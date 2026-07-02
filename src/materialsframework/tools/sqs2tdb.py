"""ATAT sqs2tdb workflow wrapper.

Drives the full SQS-to-TDB pipeline:

1. ``sqs2tdb -cp`` — stage SQS sub-directories and write ``species.in``.
2. MLIP relaxation — relax each staged structure, write ``energy`` /
   ``CONTCAR`` / ``str_relax.out`` / ``force.out`` / ``stress.out``.
3. ``sqs2tdb -fit`` — fit cluster-expansion coefficients.
4. ``sqs2tdb -tdb`` — generate the ``.tdb`` CALPHAD database file.

Requires ATAT (``sqs2tdb`` on ``$PATH``) and MaterialsFramework with at
least one registered calculator (default: ``ORBCalculator``; GRACE via
``BladeTDBGen.tdb_params["calculator"]``).
"""

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

from blade.tools.blade_sqsgen import BladeSQS

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Sqs2tdb:
    """ATAT sqs2tdb workflow driver.

    Orchestrates the four-stage SQS-to-TDB pipeline for a single chemical
    system.  Construct once, then call :meth:`fit` for each composition.

    Attributes:
        dbf (Database): Parsed TDB database object, populated after :meth:`fit`.
        species (list[str] | None): Element symbols for the active fit.
        lattices (list[str] | None): Lattice identifiers for the active fit.

    References:
        A. van de Walle et al., CALPHAD 58 (2017) 70–81.
        https://doi.org/10.1016/j.calphad.2017.05.005
    """

    VASP_WRAP = """[INCAR]
PREC = high
ISMEAR = 1
SIGMA = 0.1
NSW=41
IBRION = 2
ISIF = 3
KPPRA = 1000
USEPOT = PAWPBE
DOSTATIC
"""

    def __init__(
        self,
        md_temperature: float = 1000,
        md_pressure: float = 1,
        md_timestep: float = 1.0,
        fmax: float = 0.001,
        verbose: bool = False,
        track_trajectory: bool = True,
        calculator: BaseCalculator | BaseMDCalculator | None = None,
    ) -> None:
        """Initialize Sqs2tdb.

        Args:
            md_temperature (float, optional): MD thermostat temperature in K
                (LIQUID phase only). Defaults to 1000.
            md_pressure (float, optional): MD barostat pressure in atm
                (LIQUID phase only). Defaults to 1.
            md_timestep (float, optional): MD integration timestep in fs
                (LIQUID phase only). Defaults to 1.0.
            fmax (float, optional): Force convergence criterion in eV/Å for
                structure relaxation. Defaults to 0.001.
            verbose (bool, optional): Print calculator and command output.
                Defaults to False.
            calculator (BaseCalculator | BaseMDCalculator | None, optional):
                Energy/force/stress calculator. Defaults to
                ``ORBCalculator`` when ``None``.

        Raises:
            OSError: If ``sqs2tdb`` is not found on ``$PATH``.
        """
        if shutil.which("sqs2tdb") is None:
            raise OSError("sqs2tdb is not installed or not found in the system's PATH.")

        self.md_temperature = md_temperature
        self.md_pressure = md_pressure
        self.md_timestep = md_timestep
        self.fmax = fmax
        self.verbose = verbose
        self.track_trajectory = track_trajectory

        self._calculator = calculator

        self.species = None
        self.lattices = None
        self.level = None
        self.t_min = None
        self.t_max = None
        self.sro = None
        self.bv = None
        self.phonon = None
        self.open_calphad = None
        self.terms = None
        self.terms_in = None
        self.mult_in = None
        self.sublattice_map = None
        self.skip_existing = False
        self.path2 = None

        self.dbf = None

    def fit(
        self,
        paths,
        sqsgen_levels2,
        species: list[str],
        lattices: list[str] | None = None,
        level: int = 1,
        t_min: float = 298.15,
        t_max: float = 10000,
        sro: bool = False,
        bv: float = 5e-3,
        phonon: bool = False,
        open_calphad: bool = False,
        terms: str | None = None,
        terms_in: dict[str, str] | None = None,
        mult_in: dict[str, str] | None = None,
        sublattice_map: dict[str, dict[str, list[str]]] | None = None,
        skip_existing: bool = False,
    ) -> None:
        """Copy SQS from the database to the current directory, calculate energies, and fit a TDB model.

        Args:
            species (list): List of elements to consider (e.g., ["Al", "Ni"]).
            lattices (List[str] | None): The lattice types (e.g., ["FCC_A1", "BCC_A2"]).
            level (int): The composition mesh level (e.g., 1 for midpoints). Defaults to 1.
            t_min (float): The minimum temperature for fitting. Defaults to 298.15 K.
            t_max (float): The maximum temperature for fitting. Defaults to 10000 K.
            sro (bool): Whether to include short-range order. Defaults to False.
            bv (float): The energy bump value. Defaults to 5e-3.
            phonon (bool): Whether to include phonons for end members. Defaults to False.
            open_calphad (bool): Whether to generate an Open Calphad-compliant .tdb file. Defaults to False.
            terms (str | None): The terms to include in the model. Defaults to None.
            terms_in (dict[str, str] | None): Per-phase terms.in content keyed by lattice base name
                (e.g. ``{"HEDB1": "1,0:1,0\n2,0:1,0\n"}``). Overrides the auto-generated terms.in
                for matching phases. Phases absent from the dict use the ATAT default.
            sublattice_map (dict | None): Active species per sublattice per phase, e.g.
                ``{"FCC2": {"a": ["Cr","Hf"], "b": ["Zr","Ti"]}}``. When provided for a phase,
                species.in is overwritten after the first ``sqs2tdb -cp`` call.
            skip_existing (bool): If True, skip compositions that already have computed structures.
                Defaults to False.

        Raises:
            ValueError: If the calculator object does not implement the required properties.
            ValueError: If the lattice type is not valid.
        """
        if not all(prop in self.calculator.AVAILABLE_PROPERTIES for prop in ["energy", "forces", "stress"]):
            raise ValueError("The calculator object must have the 'energy', 'forces', and 'stress' properties implemented.")

        self.path2 = paths

        if not all(lattice in self.available_lattices for lattice in lattices):
            raise ValueError(f"Invalid lattice type. Available lattices: {self.available_lattices}")

        self.path2 = paths
        self.sqsgen_levels2 = sqsgen_levels2
        self.species = species
        self.lattices = lattices
        self.level = level
        self.t_min = t_min
        self.t_max = t_max
        self.sro = sro
        self.bv = bv
        self.phonon = phonon
        self.open_calphad = open_calphad
        self.terms = terms
        self.terms_in = terms_in
        self.mult_in = mult_in
        self.sublattice_map = sublattice_map
        self.skip_existing = skip_existing

        self._copy_sqs()
        if callable(getattr(self, "post_copy_hook", None)):
            self.post_copy_hook()
        self._fit_model()
    
        args = ["-tdb"] + (["-oc"] if open_calphad else [])
        tdb_mtimes_before = {f.name: f.stat().st_mtime for f in Path(".").glob("*.tdb")}
        self._run_command("sqs2tdb", args, timeout=None)
        tdb_mtimes_after = {f.name: f.stat().st_mtime for f in Path(".").glob("*.tdb")}

        try:
            from pycalphad import Database
        except ImportError as e:
            raise ImportError("pycalphad is required. Install it with: pip install materialsframework[calphad]") from e

        levels2_flat = (
            [item for lst in self.sqsgen_levels2.values() for item in lst]
            if isinstance(self.sqsgen_levels2, dict)
            else self.sqsgen_levels2
        )
        for i in levels2_flat:
            if i["element"] not in self.species:
                self.species.append(i["element"])

        updated_tdbs = {
            name for name, mtime in tdb_mtimes_after.items()
            if name not in tdb_mtimes_before or mtime > tdb_mtimes_before[name]
        }
        if updated_tdbs:
            tdb_path = Path(next(iter(updated_tdbs)))
        else:
            tdb_filename = "_".join(sorted([s.upper() for s in self.species])) + ".tdb"
            tdb_path = Path(tdb_filename)
        self._add_oxygen_element_if_needed(tdb_path)
        self.dbf = Database(tdb_path)

    @property
    def available_lattices(self) -> list[str]:
        """Get the list of available lattice types in the ATAT SQS database.

        Returns:
            List[str]: The list of available lattice types.
        """
        base = (Path.home() / ".atat.rc").read_text().split("=")[1].strip()
        names = {d.name for d in (Path(base) / "data" / "sqsdb").iterdir() if d.is_dir()}
        if self.path2 and Path(self.path2).exists():
            names |= {d.name for d in Path(self.path2).iterdir() if d.is_dir()}
        return sorted(names)

    @property
    def calculator(self) -> BaseCalculator | BaseMDCalculator:
        """Returns the calculator instance used for energy, force, and stress calculations.

        If the calculator instance is not already initialized, this method creates a new `ORBCalculator` instance.

        Returns:
            BaseCalculator | BaseMDCalculator: The calculator object used for energy, force, and stress calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.orb import ORBCalculator

            self._calculator = ORBCalculator()

        self._calculator.fmax = self.fmax
        self._calculator.verbose = self.verbose
        self._calculator.logfile = "-" if self.verbose else None
        self._calculator.temperature = self.md_temperature
        self._calculator.pressure = self.md_pressure
        self._calculator.timestep = self.md_timestep

        return self._calculator

    def _calculate(
        self,
        subdir: Path,
        relax: bool = True,
    ) -> None:
        """Calculate SQS energies.

        This should be run inside the relevant lattice directory.

        Args:
            subdir (Path): The path to the subdirectory containing the POSCAR file.
            relax (bool): Whether to perform relaxation. Defaults to True.
        """
        structure = Structure.from_file(subdir / "POSCAR")

        if "LIQUID" in subdir.parts:
            structure.make_supercell(2)

            self.calculator.ensemble = "npt_nose_hoover"
            res = self.calculator.run(structure=structure, steps=int(3000 / self.md_timestep))  # NPT for 3 ps

            self.calculator.ensemble = "nvt_nose_hoover"
            res = self.calculator.run(structure=res["final_structure"], steps=int(10000 / self.md_timestep))  # NVT for 10 ps

            n_last = max(1, int(0.2 * 13000))
            energy = np.mean(res["total_energy"][-n_last:])
            forces = np.mean(res["forces"][-n_last:], axis=0)
            stresses = np.mean(res["stresses"][-n_last:], axis=0)
            final_structure = res["final_structure"]

        else:
            if self.track_trajectory:
                traj_path = Path(Path(subdir).resolve()) / "relaxation_live.xyz"
                self.calculator.traj_file = str(traj_path)
                self.calculator.interval = 1
                self.calculator.verbose = True
                traj_path.touch(exist_ok=True)
            else:
                self.calculator.traj_file = None
                self.calculator.interval = 0


            res = self.calculator.relax(structure=structure) if relax else self.calculator.calculate(structure=structure)
            energy, forces, stresses, final_structure = (
                res["energy"],
                res["forces"],
                res["stress"],
                res["final_structure"],
            )

        # Write energy
        (subdir / "energy").write_text(f"{energy:.6f}")

        # Write CONTCAR
        final_structure.to(filename=str(subdir / "CONTCAR"), fmt="poscar")

        # Write str_relax.out
        with (subdir / "str_relax.out").open("w") as f:
            f.write("\n".join(" ".join(map(str, row)) for row in final_structure.lattice.matrix))
            f.write("\n1 0 0\n0 1 0\n0 0 1\n")
            f.write("\n".join(" ".join(map(str, site.frac_coords)) + " " + site.species_string for site in final_structure))

        # Write forces.out
        np.savetxt(str(subdir / "force.out"), forces, fmt="%.7e")

        # Write stress.out in Voigt notation
        if stresses.shape == (6,):
            from ase.stress import voigt_6_to_full_3x3_stress

            stresses = voigt_6_to_full_3x3_stress(stresses)
        np.savetxt(subdir / "stress.out", stresses, fmt="%.7e")

    def _run_command(self, command: str, args: list[str], cwd: Path | None = None,
                     timeout: int | None = 60) -> None:
        """Run a shell command with arguments and print stdout and stderr if verbose turned on.

        Args:
            command (str): The command to execute.
            args (list[str]): A list of arguments for the command.
            cwd (str | None): The working directory for the command.
        """
        try:
            proc = subprocess.Popen(
                [command, *args],
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
                print(f"Command timed out after {timeout}s: {command} {' '.join(args)}")
                return

            if proc.returncode not in (0, 1):
                print(f"Command failed with exit code {proc.returncode}: {(stderr or '').strip()}")
            if self.verbose:
                print("STDOUT:", (stdout or "").strip())
                print("STDERR:", (stderr or "").strip())
        except Exception as e:
            print("Unexpected error:", e)

    def _rename_files(self, sqsgen_levels2: list[dict], lattice_path: Path) -> None:
        """Append fixed-sublattice labels to species.in, mult.in, and sqs_lev dirs.

        For each fixed-sublattice entry in *sqsgen_levels2*: adds the element
        to the composition-level ``species.in``, adds the sublattice letter
        assignment to the lattice-level ``species.in``, updates ``mult.in``
        with the site count, and appends ``_<letter>_<element>=<comp>`` to
        any ``sqs_lev=*`` directory names that don't already carry that tag.

        Args:
            sqsgen_levels2 (list[dict]): Fixed-sublattice entries, each with
                keys ``"element"``, ``"letter"``, ``"compositions"``, and
                ``"count"``.
            lattice_path (Path): Path to the staged lattice directory
                (e.g. ``HEDB1_2/``).
        """
        folder = lattice_path.parent
        for i in range(len(sqsgen_levels2)):
            file_path = Path(folder) / "species.in"
            text = file_path.read_text().strip()
            if f",{sqsgen_levels2[i]['element']}" not in text:
                text = text + f",{sqsgen_levels2[i]['element']}"
            file_path.write_text(text + "\n")

            file_path = Path(lattice_path) / "species.in"
            text = file_path.read_text().strip()
            if f"{sqsgen_levels2[i]['letter']}=" not in text:
                text = text + f"\t{sqsgen_levels2[i]['letter']}={sqsgen_levels2[i]['element']}"
            file_path.write_text(text + "\n")

            mult_path = Path(lattice_path) / "mult.in"
            text = mult_path.read_text().strip()
            if re.search(rf"\b{sqsgen_levels2[i]['letter']}=", text):
                text = re.sub(rf"\b{sqsgen_levels2[i]['letter']}=\S+", f"{sqsgen_levels2[i]['letter']}=2", text)
            else:
                text += f"\t{sqsgen_levels2[i]['letter']}={sqsgen_levels2[i]['count']}"
            mult_path.write_text(text + "\n")

            for sqsdir in lattice_path.glob("sqs_lev=*/"):
                if not sqsdir.is_dir():
                    continue
                tag = f",{sqsgen_levels2[i]['letter']}_{sqsgen_levels2[i]['element']}={sqsgen_levels2[i]['compositions']}"
                if tag not in sqsdir.name:
                    new_file = sqsdir.parent / f"{sqsdir.name}{tag}"
                    sqsdir.rename(new_file)
                    print(f"Renamed {sqsdir} -> {new_file}")
                

    def _copy_sqs_folders_from_atat(self, lattice):
        src_lattice = Path(self.path2) / lattice
        dst_lattice = Path(lattice)

        if not src_lattice.exists():
            raise FileNotFoundError(f"ATAT lattice folder not found: {src_lattice}")

        dst_lattice.mkdir(exist_ok=True)

        for src in src_lattice.glob("sqsdb_*"):
            if not src.is_dir():
                continue

            new_name = src.name.replace("sqsdb_", "sqs_", 1)
            dst = dst_lattice / new_name

            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(src, dst)
            print(f"Copied {src} -> {dst}")

    def _replace_constant_elements_in_sqs_files(self, lattice_path):
        lattice_path = Path(lattice_path)

        constant_items = (
            [item for lst in self.sqsgen_levels2.values() for item in lst]
            if isinstance(self.sqsgen_levels2, dict)
            else self.sqsgen_levels2
        )

        if not constant_items:
            return

        for bestsqs_path in lattice_path.glob("sqs_lev=*/bestsqs.out"):
            text = bestsqs_path.read_text()

            for item in constant_items:
                print(item)
                element = item["element"]
                letter = item["letter"]
                replacement = f"{letter}_A,{letter}_B"
                print(element, letter, replacement)

                text = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(element)}(?![A-Za-z0-9_])",
                    replacement,
                    text,
                )

            bestsqs_path.write_text(text)
            print(f"Updated constants in {bestsqs_path}")

        for rndstr_path in lattice_path.glob("sqs_lev=*/rndstr.in"):
            text = rndstr_path.read_text()

            for item in constant_items:
                element = item["element"]
                letter = item["letter"]
                replacement = f"{letter}_A=1.0,{letter}_B=0.0"

                text = re.sub(
                    rf"(?<![A-Za-z0-9_=]){re.escape(element)}(?![A-Za-z0-9_=])",
                    replacement,
                    text,
                )

            rndstr_path.write_text(text)
            print(f"Updated constants in {rndstr_path}")

    def _add_oxygen_element_if_needed(self, tdb_path):
        tdb_path = Path(tdb_path)
        text = tdb_path.read_text()

        # Only patch if O is actually in the species list
        if "O" not in [s.upper() for s in self.species]:
            return

        # Do not add duplicate ELEMENT O lines
        if re.search(r"^ELEMENT\s+O\s+", text, flags=re.MULTILINE):
            return

        oxygen_line = "ELEMENT O 1/2_MOLE_O2(G) 0.015999 4341.0 102.515 !"

        lines = text.splitlines()

        # Find the last ELEMENT line
        insert_at = None
        for idx, line in enumerate(lines):
            if line.strip().upper().startswith("ELEMENT "):
                insert_at = idx + 1

        if insert_at is None:
            raise ValueError(f"No ELEMENT lines found in {tdb_path}")

        lines.insert(insert_at, oxygen_line)

        tdb_path.write_text("\n".join(lines) + "\n")
        print(f"Added ELEMENT O line to {tdb_path}")

    def _copy_sqs(self) -> None:
        """Stage SQS directories, write species.in, and run MLIP relaxations.

        For each lattice in :attr:`lattices`:

        1. Run ``sqs2tdb -cp`` twice — first call creates ``species.in``,
           second call populates ``sqs_lev=*`` sub-directories.
        2. If :attr:`sublattice_map` provides an entry for this lattice,
           overwrite the lattice-level ``species.in`` with per-sublattice
           assignments and extend the composition-level ``species.in`` with
           any elements not already listed.
        3. Append fixed-sublattice labels via :meth:`_rename_files`.
        4. Relax every structure that has a ``wait`` file via
           :meth:`_calculate`.
        5. Optionally run phonon calculations for end-members when
           :attr:`phonon` is ``True``.
        """
        species_str = ",".join(self.species)

        for lattice in self.lattices:
            lattice_path = Path(lattice)

            src_lattice = Path(self.path2) / lattice
            sqsdb_count = sum(1 for d in src_lattice.glob("sqsdb_lev=*") if d.is_dir()) if src_lattice.exists() else 0

            if isinstance(self.sqsgen_levels2, dict):
                lattice_base = lattice.rsplit("_", 1)[0]
                levels2 = self.sqsgen_levels2.get(lattice_base, [])
            else:
                levels2 = self.sqsgen_levels2
            fixed_letters = {entry["letter"] for entry in levels2}

            if sqsdb_count > 0 and src_lattice.exists():
                sample_dirs = [d.name for d in src_lattice.glob("sqsdb_lev=*") if d.is_dir()]
                var_letters: set[str] = set()
                for dname in sample_dirs:
                    for m in re.finditer(r"_([a-z])_[A-Z]", dname):
                        letter = m.group(1)
                        if letter not in fixed_letters:
                            var_letters.add(letter)
                if len(var_letters) > 1 and len(self.species) < len(var_letters):
                    print(
                        f"Warning: {lattice} has {len(var_letters)} variable sublattices "
                        f"({', '.join(sorted(var_letters))}) but only {len(self.species)} species. "
                        f"ATAT's sqs2tdb -cp cannot map multi-sublattice binary compositions — "
                        f"only endmember levels will be copied."
                    )

            # First -cp: creates species.in and exits code 1
            self._run_command(
                "sqs2tdb",
                ["-cp", f"-l={lattice}", f"-lv={self.level}", f"-sp={species_str}"],
                timeout=None,
            )

            # Override species.in for custom multi-sublattice phases
            lattice_base = lattice.rsplit("_", 1)[0]
            smap = (self.sublattice_map or {}).get(lattice_base)
            if smap and lattice_path.exists():
                # Intersect each sublattice's pool with the actual composition species
                comp_set = set(self.species)
                filtered = {
                    letter: [el for el in elems if el in comp_set]
                    for letter, elems in smap.items()
                    if letter != "Constant"
                }
                lines = [
                    f"{letter}={','.join(sorted(active))}"
                    for letter, active in sorted(filtered.items())
                    if active  # skip sublattices with no matching species
                ]
                (lattice_path / "species.in").write_text("\t".join(lines) + "\n")
                comp_species_in = lattice_path.parent / "species.in"
                if comp_species_in.exists():
                    existing = comp_species_in.read_text().strip()
                    existing_els = set(existing.split(","))
                    for el in sorted({el for els in smap.values() for el in els}):
                        if el not in existing_els:
                            existing = existing + f",{el}"
                            existing_els.add(el)
                    comp_species_in.write_text(existing + "\n")

            print('sqs2tdb -cp completed for lattice:', lattice)

            # Second -cp: reads species.in and creates sqs_lev dirs.
            # When a fixed composition is requested (dir_filter set), ATAT still generates
            # all permutations of element-fraction assignments. We run -cp in the background
            # and monitor for the matching dir. Non-matching dirs are deleted immediately
            # once ATAT finishes writing them (signalled by the 'wait' file). Once the match
            # is found and the next dir confirms it is fully written, -cp is killed early to
            # avoid creating all remaining permutations.
            if callable(getattr(self, "dir_filter", None)):
                import time as _time
                print(f"[dir_filter] Running sqs2tdb -cp in background; will delete non-matching dirs as they appear and stop early once match is found.")
                # No stdout/stderr pipes so ATAT doesn't block on buffer fills
                proc = subprocess.Popen(
                    ["sqs2tdb", "-cp", f"-l={lattice}", f"-lv={self.level}", f"-sp={species_str}"],
                    start_new_session=True,
                )
                # Count expected matches: one per non-blank sqsgen.in line
                sqsgen_path = Path(self.path2) / lattice / "sqsgen.in"
                expected = sum(
                    1 for ln in sqsgen_path.read_text().splitlines() if ln.strip()
                ) if sqsgen_path.exists() else 1

                def _kill_proc():
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        pass

                found_count = 0
                seen: set[str] = set()
                try:
                    while True:
                        for d in list(lattice_path.glob("sqs_lev=*")):
                            if not d.is_dir() or d.name in seen:
                                continue
                            # Only act once ATAT finishes writing (wait file = last step)
                            if not (d / "wait").exists():
                                continue
                            seen.add(d.name)
                            if self.dir_filter(d):
                                found_count += 1
                                print(f"[dir_filter] Match {found_count}/{expected}: {d.name}")
                                if found_count >= expected:
                                    _kill_proc()
                                    print(f"[dir_filter] All {expected} matches found — killed sqs2tdb -cp")
                            else:
                                shutil.rmtree(d, ignore_errors=True)
                                print(f"[dir_filter] Deleted non-matching dir: {d.name}")
                        if proc.poll() is not None:
                            break
                        _time.sleep(0.05)
                except KeyboardInterrupt:
                    _kill_proc()
                    raise
                proc.wait()
                # Final sweep for any dirs created after kill
                for d in list(lattice_path.glob("sqs_lev=*")):
                    if d.is_dir() and not self.dir_filter(d):
                        shutil.rmtree(d, ignore_errors=True)
                        print(f"[dir_filter] Final sweep deleted: {d.name}")
                if found_count == 0:
                    print("[dir_filter] Warning: no matching dir found — all dirs processed")
            else:
                self._run_command(
                    "sqs2tdb",
                    ["-cp", f"-l={lattice}", f"-lv={self.level}", f"-sp={species_str}"],
                    timeout=None,
                )

            print('sqs2tdb -cp completed for lattice:', lattice)

            # Build POSCAR from str.in for local sqs_lev dirs that don't have one yet.
            # str.in is ATAT format with generic labels (a_A, a_B, b_A, B, C, ...).
            # Map generic labels to actual elements using the dir name, then write POSCAR.
            for local_dir in lattice_path.glob("sqs_lev=*"):
                if not local_dir.is_dir() or (local_dir / "POSCAR").exists():
                    continue
                str_in = local_dir / "str.in"
                if not str_in.exists():
                    continue
                # Build generic-label → element mapping from dir name
                # e.g. a_Cr=0.875,a_Hf=0.125,b_B=1.0 → {a_A:Cr, a_B:Hf, b_A:B} (sorted desc frac)
                el_fracs: list[tuple[str, str, float]] = []
                for m in re.finditer(r'([a-z])_([A-Z][a-z]?)=([\d.]+)', local_dir.name):
                    letter, el, frac = m.group(1), m.group(2), float(m.group(3))
                    el_fracs.append((letter, el, frac))
                # Group by sublattice letter, sort each group descending by frac → A,B,C...
                from collections import defaultdict
                by_letter: dict = defaultdict(list)
                for letter, el, frac in el_fracs:
                    by_letter[letter].append((frac, el))
                label_map: dict[str, str] = {}
                abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for letter, pairs in by_letter.items():
                    for idx, (_, el) in enumerate(sorted(pairs, reverse=True)):
                        label_map[f"{letter}_{abc[idx]}"] = el
                # Parse str.in: lines 0-2 prim lattice, 3-5 supercell, 6+ coords+label
                try:
                    lines = str_in.read_text().splitlines()
                    prim = [[float(x) for x in lines[i].split()] for i in range(3)]
                    sc   = [[float(x) for x in lines[i].split()] for i in range(3, 6)]
                    import numpy as _np
                    P = _np.array(prim)
                    S = _np.array(sc)
                    cart_lat = S @ P
                    inv_lat  = _np.linalg.inv(cart_lat)
                    species_list, frac_list = [], []
                    for line in lines[6:]:
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        # str.in positions are fractional in primitive lattice → convert to Cartesian
                        prim_frac = _np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                        cart = prim_frac @ P
                        raw_label = parts[3]
                        el = label_map.get(raw_label, raw_label)
                        frac_list.append(cart @ inv_lat)
                        species_list.append(el)
                    from pymatgen.core import Lattice as _Lat, Structure as _Str
                    structure = _Str(_Lat(cart_lat), species_list, frac_list, coords_are_cartesian=False)
                    structure.to(filename=str(local_dir / "POSCAR"), fmt="poscar")
                    print(f"Built POSCAR from str.in: {local_dir.name}")
                except Exception as e:
                    print(f"Failed to build POSCAR from str.in for {local_dir.name}: {e}")

            if len(self.species) == 1:
                self._copy_sqs_folders_from_atat(lattice)

            if isinstance(self.sqsgen_levels2, dict):
                levels2 = self.sqsgen_levels2.get(lattice_base, [])
            else:
                levels2 = self.sqsgen_levels2
            self._rename_files(levels2, lattice_path)

            for wait_file in lattice_path.glob("*/wait"):
                subdir = wait_file.parent
                if not subdir.exists():
                    continue
                if callable(getattr(self, "dir_filter", None)):
                    if not self.dir_filter(subdir):
                        shutil.rmtree(subdir, ignore_errors=True)
                        continue
                if self.skip_existing and (subdir / "energy").exists():
                    wait_file.unlink()
                    continue
                (subdir / "vasp.wrap").write_text(self.VASP_WRAP)
                poscar = subdir / "POSCAR"
                if not poscar.exists() or poscar.stat().st_size == 0:
                    # POSCAR not created by _poscar_from_bestsqs — try runstruct_vasp
                    self._run_command("runstruct_vasp", ["-nr"], cwd=subdir)
                # Validate POSCAR has at least 6 lines (comment, scale, 3 lattice, species/counts)
                if not poscar.exists():
                    print(f"Skipping {subdir.name}: POSCAR missing")
                    continue
                try:
                    poscar_lines = poscar.read_text().splitlines()
                    if len(poscar_lines) < 6:
                        print(f"Skipping {subdir.name}: POSCAR malformed ({len(poscar_lines)} lines)")
                        continue
                except Exception:
                    print(f"Skipping {subdir.name}: POSCAR unreadable")
                    continue
                try:
                    self._calculate(subdir)
                except Exception as e:
                    print(f"Skipping {subdir.name}: _calculate failed ({e})")
                    continue
                if wait_file.exists():
                    wait_file.unlink()

            if self.phonon:
                for endmember in lattice_path.glob("*/endmem"):
                    self._run_command(
                        "fitfc",
                        ["-si=str_relax.out", "-ernn=3", "-ns=1", "-nrr"],
                        cwd=endmember.parent,
                    )

                for wait_file in lattice_path.rglob("wait"):
                    subdir = wait_file.parent
                    (subdir / "vasp.wrap").write_text(self.VASP_WRAP)
                    self._run_command("runstruct_vasp", ["-nr"], cwd=subdir)
                    self._calculate(subdir, relax=False)
                    wait_file.unlink()

                for endmember in lattice_path.glob("*/endmem"):
                    subdir = endmember.parent
                    self._run_command("fitfc", ["-si=str_relax.out", "-f", "-frnn=1.5"], cwd=subdir)
                    self._run_command("robustrelax_vasp", ["-vib"], cwd=subdir)

    def _fit_model(self) -> None:
        """Run ``sqs2tdb -fit`` twice for each lattice to fit CALPHAD coefficients.

        First pass generates the default ``terms.in`` / ``mult.in``.  Then
        those files are optionally overridden by :attr:`terms_in` /
        :attr:`mult_in` before a second ``sqs2tdb -fit`` call uses the
        updated inputs.  Falls back to built-in defaults if neither
        :attr:`terms_in` nor :attr:`terms` is set.
        """
        for lattice in self.lattices:
            args = ["-fit", f"-Tl={self.t_min}", f"-Tu={self.t_max}"]
            if self.bv:
                args.append(f"-bv={self.bv}")
            if self.sro:
                args.append("-sro")

            lattice_path = Path(lattice).resolve()
            self._run_command("sqs2tdb", args, cwd=lattice_path, timeout=None)

            lattice_base = lattice.rsplit("_", 1)[0]
            if self.terms_in and lattice_base in self.terms_in:
                terms_content = self.terms_in[lattice_base]
            elif self.terms:
                terms_content = self.terms
            else:
                terms_content = (
                    "1,0\n2,1" if lattice in ["BCC_A2", "FCC_A1", "HCP_A3"] else "1,0:1,0\n2,0:1,0\n"
                )

            (lattice_path / "terms.in").write_text(terms_content)

            if self.mult_in and lattice_base in self.mult_in:
                (lattice_path / "mult.in").write_text(self.mult_in[lattice_base])

            self._run_command("sqs2tdb", args, cwd=lattice_path, timeout=None)
            
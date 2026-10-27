"""Translates a title's eboot to arm64 and packages it as a macOS application that runs without Rosetta.

    python3 tools/translator/package_native_app.py --eboot <eboot.bin|eboot.elf> --game <dir> \\
        --libs <build-arm64/core/libs/libs> --vulkan <Vulkan SDK macOS dir> [--translator <build-translator>] [--tier quick|full] [--jobs N|max] \\
        [--module-dir <dir>]... [--skip-sce-module] <Title.app>

--eboot is the title's executable, a decrypted SELF (unwrapped here) or a plain ELF. --game is the title's
folder (sce_sys and its data), copied as app0. --libs is the arm64 prx directory and --vulkan the macOS
directory of a Vulkan SDK. aps5-translator lifts the eboot to an arm64 object; imports that no arm64
library exports are linked to stubs that name the missing NID and abort. The bundle has the same layout
as tools/package_macos_app.py makes for the Rosetta mode, without the x86 game and MoltenVK patches.

--tier quick (the default) translates in minutes: code the title has not run yet is compiled quickly and
counts how often it runs, and packaging the title again after playing optimizes what the profile shows
hot. --tier full optimizes everything up front. --jobs sets the translator threads, all cores by default;
max also lifts the memory budget that otherwise keeps a translation within the machine's RAM.

The guest modules the title bundles (ELF prx files in sce_module, sce_modules and prx) are translated too,
each to an arm64 dylib next to it named <module>.guest.prx, which is where libkernel's loader looks, as for
the relinker's macOS mode. The executable links them, so they load and initialize at start, dependencies
first. --module-dir names another folder of the title, such as Media/Plugins, whose modules are only
loaded when the title asks for them (sceKernelLoadStartModule). A module's exports win over the arm64
libraries, except the zero-only exports of a stub library. --skip-sce-module leaves the bundled modules out."""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(TOOLS))
import package_macos_app

LLVM = Path("/opt/homebrew/opt/llvm/bin")
NID = re.compile(r"_[A-Za-z0-9+-]{11}")
MODULE_DIRECTORIES = ("sce_module", "sce_modules", "prx")


def elf_of(eboot, work):
    if eboot.read_bytes()[:4] == b"\x7fELF":
        return eboot
    elf = work / "eboot.elf"
    subprocess.run([sys.executable, str(TOOLS / "unwrap_self.py"), str(eboot), str(elf)], check=True)
    return elf


def exported_names(libraries):
    """The names dyld binds against. The NID patcher renames exports in the export trie only, so the
    symbol table (nm) still shows the original names."""
    names = set()
    for library in libraries:
        listing = subprocess.run([str(LLVM / "llvm-objdump"), "--macho", "--exports-trie", str(library)], capture_output=True, text=True).stdout
        for line in listing.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].startswith("0x"):
                names.add(fields[1])
    return names


def dynamic_names(elf):
    """The DT_NEEDED names and the DT_SONAME of an ELF."""
    listing = subprocess.run([str(LLVM / "llvm-readelf"), "-d", str(elf)], capture_output=True, text=True, check=True).stdout
    soname = re.findall(r"Library soname: \[(.+?)\]", listing)
    return re.findall(r"Shared library: \[(.+?)\]", listing), soname[0] if soname else None


def needed_libraries(elf, libs):
    """The arm64 libraries an image links after its guest modules, which win the names both export."""
    needed, _ = dynamic_names(elf)
    return [libs / name for name in dict.fromkeys(["libkernel.prx", "libc.prx", *needed]) if (libs / name).is_file()]


def write_stubs(missing, path, hidden=False):
    with path.open("w") as out:
        out.write("#include <stdio.h>\n#include <stdlib.h>\n")
        visibility = '__attribute__((visibility("hidden"))) ' if hidden else ""
        for index, name in enumerate(missing):
            out.write(f'{visibility}void stub{index}(void) __asm__("{name}");\n')
            out.write(f'void stub{index}(void) {{ fprintf(stderr, "[native] missing import {name[1:]}\\n"); abort(); }}\n')


def symbols(path, *flags):
    return set(subprocess.run([str(LLVM / "llvm-nm"), *flags, "--format=just-symbols", str(path)], capture_output=True, text=True, check=True).stdout.split())


def translate(elf, translator, guest, tier, jobs, host_exports=None):
    environment = dict(os.environ, APS5_TRANSLATOR_TIER=tier, **({"APS5_TRANSLATOR_JOBS": jobs} if jobs else {}))
    if host_exports is not None:
        environment["APS5_TRANSLATOR_HOST_EXPORTS"] = str(host_exports)
    translated = subprocess.run([str(translator / "aps5-translator"), str(elf), str(guest)], capture_output=True, text=True, env=environment)
    print(f"{elf.name}: {translated.stderr.strip()}")
    if translated.returncode != 0:
        raise RuntimeError(f"translation of {elf} failed")


def stub_missing(guest, exported, stubs, label, hidden=False):
    """Links the NID imports nothing exports to stubs that name them and abort. Plain names such as
    ___divti3 come from libSystem; only NIDs get stubs."""
    missing = sorted(name for name in symbols(guest, "-u") - exported if NID.fullmatch(name))
    if missing:
        print(f"{label}: {len(missing)} imports have no arm64 export and abort when called: {' '.join(name[1:] for name in missing)}")
    write_stubs(missing, stubs.with_suffix(".c"), hidden)
    subprocess.run(["clang", "-arch", "arm64", "-c", str(stubs.with_suffix(".c")), "-o", str(stubs)], check=True)


class Module:
    def __init__(self, path, game, deferred):
        self.path = path
        self.relative = path.relative_to(game)
        self.deferred = deferred
        self.needed, self.soname = dynamic_names(path)
        self.output = None

    def names(self):
        return {self.path.name, *([self.soname] if self.soname else [])}

    def stems(self):
        """Needed names match a module by stem, so libfoo.prx finds libfoo.sprx."""
        return {Path(name).stem for name in self.names()}


def bundled_modules(game, module_directories):
    """The title's guest modules, as the relinker's macOS mode finds them: sce_module (or sce_modules)
    and prx load at start, --module-dir folders on request."""
    directories = [(game / name, False) for name in MODULE_DIRECTORIES if (game / name).is_dir()]
    directories += [(game / name, True) for name in module_directories]
    modules = []
    for directory, deferred in directories:
        if not directory.is_dir():
            raise RuntimeError(f"Guest module path is not a directory: {directory}")
        for path in sorted(directory.iterdir()):
            if path.name.endswith(".guest.prx") or not path.is_file():
                continue
            header = path.read_bytes()[:20]
            magic = header[:4]
            if magic == b"\x7fELF" and header[18:20] == b"\x3e\x00":
                modules.append(Module(path, game, deferred))
            elif magic in (b"\x4f\x15\x3d\x1d", b"\x54\x14\xf5\xee"):
                print(f"Skipping SELF container: {path.relative_to(game)} (unwrap it before packaging)")
    return modules


def initialization_order(modules):
    """Dependencies first, so each module links against the guest modules it needs."""
    by_stem = {stem: module for module in modules for stem in module.stems()}
    order, state = [], {}

    def visit(module):
        if state.get(id(module)) == 1:
            raise RuntimeError(f"Cyclic guest initialization dependency: {module.relative}")
        if state.get(id(module)) == 2:
            return
        state[id(module)] = 1
        for name in module.needed:
            provider = by_stem.get(Path(name).stem)
            if provider is not None and provider is not module:
                visit(provider)
        state[id(module)] = 2
        order.append(module)

    for module in modules:
        visit(module)
    return order


def runtime_imports(*objects):
    """The translator runtime functions a module calls; the executable defines them."""
    undefined = set().union(*(symbols(path, "-u") for path in objects))
    defined = set().union(*(symbols(path, "--defined-only") for path in objects))
    return sorted(name for name in undefined - defined if name.startswith("_aps5_"))


def build_modules(modules, libs, translator, work, tier, jobs, host_exports, output):
    """Translates and links every module into output/app0; returns the start modules' dylibs."""
    host_names = work / "host-exports.txt"
    host_names.write_text("".join(name[1:] + "\n" for name in sorted(host_exports)))
    directories = sorted({str(module.relative.parent) for module in modules})
    runtime = translator / "libaps5-translator-module.a"
    exported = set(host_exports)
    linked = []
    for index, module in enumerate(initialization_order(modules)):
        scratch = work / f"module{index}"
        scratch.mkdir()
        guest = scratch / "guest.o"
        translate(module.path, translator, guest, tier, jobs, host_names)
        stub_missing(guest, exported, scratch / "stubs.o", str(module.relative), hidden=True)
        module.output = output / "app0" / module.relative.parent / (module.path.name + ".guest.prx")
        module.output.parent.mkdir(parents=True, exist_ok=True)
        dependencies = [other.output for other in linked if not other.deferred or module.deferred]
        rpaths = ["@loader_path", "@executable_path/libs", *(f"@executable_path/app0/{directory}" for directory in directories)]
        subprocess.run(["clang++", "-arch", "arm64", "-dynamiclib", str(guest), str(scratch / "stubs.o"), f"-Wl,-force_load,{runtime}",
                        *map(str, dependencies), *map(str, needed_libraries(module.path, libs)),
                        *(f"-Wl,-U,{name}" for name in runtime_imports(guest, runtime)), f"-Wl,-install_name,@rpath/{module.output.name}",
                        *(f"-Wl,-rpath,{rpath}" for rpath in rpaths), "-o", str(module.output)], check=True)
        exported |= exported_names([module.output])
        linked.append(module)
    return [module.output for module in linked if not module.deferred], directories, exported


def build_executable(elf, libs, translator, work, tier, jobs, modules=()):
    output = work / "relinked"
    output.mkdir()
    host_exports = exported_names(sorted(libs.glob("*.prx")))
    start, directories, exported = build_modules(modules, libs, translator, work, tier, jobs, host_exports, output) if modules else ([], [], host_exports)
    guest = work / "guest.o"
    translate(elf, translator, guest, tier, jobs)
    stub_missing(guest, exported, work / "stubs.o", "eboot")
    rpaths = ["@executable_path/libs", *(f"@executable_path/app0/{directory}" for directory in directories)]
    subprocess.run(["clang++", "-arch", "arm64", str(guest), str(work / "stubs.o"), str(translator / "libaps5-translator-runtime.a"),
                    *map(str, start), *map(str, needed_libraries(elf, libs)), *(f"-Wl,-rpath,{rpath}" for rpath in rpaths), "-o", str(output / "eboot")], check=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eboot", required=True, type=Path)
    parser.add_argument("--game", required=True, type=Path)
    parser.add_argument("--libs", required=True, type=Path)
    parser.add_argument("--vulkan", required=True, type=Path)
    parser.add_argument("--translator", type=Path, default=Path("build-translator"))
    parser.add_argument("--tier", choices=["quick", "full"], default="quick")
    parser.add_argument("--jobs")
    parser.add_argument("--module-dir", action="append", default=[], help="a folder of the title whose modules load on request, such as Media/Plugins")
    parser.add_argument("--skip-sce-module", action="store_true", help="do not translate the modules the title bundles")
    parser.add_argument("bundle", type=Path)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="anyps5-native-") as directory:
        work = Path(directory)
        modules = [] if arguments.skip_sce_module else bundled_modules(arguments.game.resolve(), arguments.module_dir)
        relinked = build_executable(elf_of(arguments.eboot, work), arguments.libs, arguments.translator, work, arguments.tier, arguments.jobs, modules)
        name, identifier = package_macos_app.package(relinked, arguments.game, arguments.libs, arguments.vulkan, None, arguments.bundle, patch=False)
    print(f"{arguments.bundle}: {name} ({identifier}), arm64")


if __name__ == "__main__":
    main()

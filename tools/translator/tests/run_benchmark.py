"""Times translated guest code: builds bench_cpu.c for the guest, translates it once per configuration and runs it.

usage: run_benchmark.py <build-translator> <arm64 libs directory> [NAME=VAR=VALUE,VAR=VALUE ...]

Each extra argument names a configuration and the translator environment it sets, for example
quick=APS5_TRANSLATOR_TIER=quick. The baseline configuration always runs first. A quick tier
configuration runs twice: cold, which writes the profile, then translated again with that profile."""
import os, pathlib, shutil, statistics, subprocess, sys, tempfile, time

import run_native_fixtures as fixtures


def configurations(arguments):
    result = [("baseline", {})]
    for argument in arguments:
        name, _, settings = argument.partition("=")
        result.append((name, dict(setting.split("=", 1) for setting in settings.split(",") if setting)))
    return result


def main():
    build, libraries = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    work = pathlib.Path(tempfile.mkdtemp(prefix="anyps5-bench-"))
    guest = work / "guest"
    guest.mkdir()
    fixtures.compiled_guest(guest, "bench_cpu.c", ["-O2"])
    outputs = set()

    def measure(name, directory, environment):
        started = time.monotonic()
        subprocess.run([str(build / "aps5-translator"), str(guest / "eboot.elf"), str(directory / "guest.o")], check=True, env=environment, capture_output=True)
        translated = time.monotonic() - started
        subprocess.run(["clang++", "-arch", "arm64", str(directory / "guest.o"), str(build / "libaps5-translator-runtime.a"), str(directory / "libs/libc.prx"),
                        "-Wl,-rpath,@executable_path/libs", "-o", str(directory / "bench")], check=True)
        times = []
        for _ in range(5):
            started = time.monotonic()
            executed = subprocess.run([str(directory / "bench")], capture_output=True, text=True, timeout=600, cwd=directory)
            times.append(time.monotonic() - started)
            outputs.add(executed.stdout.strip())
        text = subprocess.run(["size", "-m", str(directory / "guest.o")], capture_output=True, text=True).stdout
        print(f"{name}: run median {statistics.median(times):.3f} s (min {min(times):.3f}), translate {translated:.2f} s, exit {executed.returncode}, output {executed.stdout.strip()}")
        if text:
            print("  " + " ".join(line.strip() for line in text.splitlines() if "__text" in line))

    for name, settings in configurations(sys.argv[3:]):
        directory = work / name
        (directory / "libs").mkdir(parents=True)
        shutil.copy2(libraries / "libc.prx", directory / "libs" / "libc.prx")
        environment = dict(os.environ, APS5_TRANSLATOR_CACHE="off", **settings)
        quick = settings.get("APS5_TRANSLATOR_TIER") == "quick"
        if quick:
            environment["APS5_TRANSLATOR_PROFILE"] = str(directory / "profile.txt")
        measure(name, directory, environment)
        if quick:
            measure(name + " with profile", directory, environment)
    if len(outputs) != 1:
        raise SystemExit(f"configurations disagree: {sorted(outputs)}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

# AnyPS5 Launcher

**AnyPS5.app**, the launcher of [AnyPS5-ARM](https://github.com/utkuhalis/AnyPS5-ARM): a home screen laid out like the PS5's, in SwiftUI, for macOS 14 and newer on Apple silicon.

![The launcher's home screen](https://raw.githubusercontent.com/utkuhalis/AnyPS5-ARM/main/docs/screenshots/launcher-home.jpg)

This branch holds the launcher alone. The whole project, with the engine the launcher runs games with and the phone app, is on [`main`](https://github.com/utkuhalis/AnyPS5-ARM/tree/main); releases are made from there.

## Windows and Linux

This launcher is for macOS. On **Windows and Linux**, use [**AnyPS5 Launcher** by AlxcBTW](https://github.com/AlxcBTW/anyps5-launcher), a port of this launcher for [AnyPS5](https://github.com/boykopovar/AnyPS5) on those systems: the same PS5-style home screen, with unlocking and starting games, per-game controls, trophies and Discord. Downloads are on its [releases page](https://github.com/AlxcBTW/anyps5-launcher/releases/latest).

## What it does

- **Finds the games** in a folder of PS5 dumps, with their icons, backgrounds, music, sizes and trophy counts.
- **Unlocks them:** the engine's relinker turns each game into a Mac app, kept in the games folder (`.AnyPS5/Games`), with the game's own files and saves beside it (`.AnyPS5/Data`) so macOS checks the app only once.
- **Starts them**, with a start screen until the game's first window shows, and says in plain words why a game stopped when it ends on an error.
- **Per game:** controls (keyboard and mouse to pad), resolution, language, deadzones, the mouse as the camera, the shader cache, and sharing all of it as one file.
- **Screenshots and saves:** a gallery of each game's F12 shots, and backing up and restoring its saves and trophies.
- **Trophies:** the list as the console shows it, and a card over the game when one is earned.
- **Music:** the selected game's own theme song fades in, like on the PS5, with volumes for it and the interface sounds.
- **Updates:** it tells you when a new AnyPS5 is out and replaces itself with it from Settings → About.
- **Getting around:** arrow keys, Return, Esc and O, or a controller; L1 / R1 (Q / E) jump between games, also inside a game's panels.
- **Discord:** the game you play on your profile, as itself when Discord knows it.
- **The phone as a controller:** a server for AnyPS5 Remote (branch [`mobile-controller`](https://github.com/utkuhalis/AnyPS5-ARM/tree/mobile-controller)): pairing, the game list, moving around the launcher, starting and stopping games, and relaying the pad to the game.
- English and Turkish; English by default.

## Layout

| Folder | What it is |
|---|---|
| `Sources/AnyPS5` | The app: screens, panels, input, sounds, the headless preview renderer. |
| `Sources/AnyPS5Core` | What the app and the tool share: the library scan, installing, game settings, trophies, diagnosis, Discord, the phone server. |
| `Sources/anyps5-tool` | A command-line tool on the same core: `scan`, `install`, `unwrap`, and `remote` (the phone server, with a pretend library for testing). |
| `Tests/AnyPS5CoreTests` | The core's tests. |
| `icon.swift` | Draws the app icon. |

## Building

```sh
swift build            # debug build of the app and the tool
swift test             # the core's tests
```

The app needs the engine's toolkit (the relinker, the system libraries, MoltenVK) to unlock and start games. A full AnyPS5.app with the toolkit inside is built from `main`, with `tools/macos/build_launcher_app.sh`.

### Screens without a window

```sh
.build/debug/AnyPS5 --render-preview <output folder> <games folder>
```

draws every screen into PNG files without opening a window. `ANYPS5_PREVIEW_SELECT=<title ID>` picks the game shown and `ANYPS5_PREVIEW_LANGUAGE=tr` draws them in Turkish.

## License

[LICENSE](LICENSE)

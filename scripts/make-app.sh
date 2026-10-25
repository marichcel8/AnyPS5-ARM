#!/bin/sh
# Wraps the launcher binary in AnyPS5.app. Run from a terminal as a bare binary, macOS gives
# the process no key window and no keyboard focus; as an app bundle with an Info.plist it
# activates like any app. Usage: scripts/make-app.sh [debug|release] [output.app]
set -e
cd "$(dirname "$0")/.."
config="${1:-release}"
app="${2:-.build/AnyPS5.app}"
swift build -c "$config" --arch arm64 --product AnyPS5
bin="$(swift build -c "$config" --arch arm64 --show-bin-path)"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$app/Contents/Frameworks"
cp "$bin/AnyPS5" "$app/Contents/MacOS/AnyPS5"
[ -d Fonts ] && cp -R Fonts "$app/Contents/Resources/Fonts"
for framework in "$bin"/*.framework; do
    [ -d "$framework" ] && cp -R "$framework" "$app/Contents/Frameworks/"
done
for bundle in "$bin"/*.bundle; do
    [ -d "$bundle" ] && cp -R "$bundle" "$app/Contents/Resources/"
done
if ! otool -l "$app/Contents/MacOS/AnyPS5" | grep -q "@executable_path/../Frameworks"; then
    install_name_tool -add_rpath @executable_path/../Frameworks "$app/Contents/MacOS/AnyPS5"
fi
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>AnyPS5</string>
    <key>CFBundleDisplayName</key><string>AnyPS5</string>
    <key>CFBundleIdentifier</key><string>org.anyps5.launcher</string>
    <key>CFBundleExecutable</key><string>AnyPS5</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.2.18</string>
    <key>CFBundleVersion</key><string>0.2.18</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.games</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <key>NSMicrophoneUsageDescription</key><string>Voice chat in AnyTogether parties.</string>
</dict>
</plist>
PLIST
codesign --force --deep --sign - --entitlements AnyPS5.entitlements "$app"
echo "$app"

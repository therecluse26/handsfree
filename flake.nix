{
  description = "handsfree - hands-free webcam mouse for Linux (dev shell)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (pkgs:
        let
          # Shared libraries that the prebuilt PyPI wheels (mediapipe, opencv,
          # PySide6) expect from a normal FHS distro.
          runtimeLibs = with pkgs; [
            stdenv.cc.cc.lib
            zlib
            zstd
            glib
            libGL
            libglvnd
            libdrm
            mesa
            freetype
            fontconfig
            dbus
            libxkbcommon
            wayland
            libx11
            libxcb
            libxext
            libxrender
            libxi
            libsm
            libice
            xcb-util-cursor
            libxcb-util
            libxcb-image
            libxcb-keysyms
            libxcb-render-util
            libxcb-wm
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [ pkgs.uv pkgs.v4l-utils ];
            # uv downloads a standalone CPython. LD_LIBRARY_PATH gives the
            # wheels their shared libraries; SSL_CERT_FILE lets the one-time
            # model download verify HTTPS.
            UV_PYTHON_PREFERENCE = "only-managed";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
          };
        });
    };
}

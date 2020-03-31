let
  pkgs = import <nixpkgs> {};
in
with pkgs;
mkShell {
  buildInputs = [
    nixfmt
    (python3.withPackages (ps: with ps; [numpy setuptools yapf matplotlib, numba]))
  ];
}

{
  description = "noisy-state-lib flake ";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

      in
      {
        defaultPackage = pkgs.python3Packages.buildPythonPackage rec {
          name = "noisy-state-lib";
          version = "0.0.1";
          src = ./.;
          propagatedBuildInputs = with pkgs.python3Packages; [
            numpy
            matplotlib
            numba
          ];
        };
      });
}

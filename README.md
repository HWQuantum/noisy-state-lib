# noisy-state-lib
A storage place for the functions to do with noisy states, etc...


## Using with Nix
Here's an example of a minimal `shell.nix` file to set up an environment with this library

Make sure you give the proper revision to the rev variable in fetchGit.
```nix
let
  pkgs = import <nixpkgs> {};
  mode_sorting = with pkgs;
    python3.pkgs.buildPythonPackage rec {
      pname = "noisy-state-lib";
      version = "0.0.1";
      src = builtins.fetchGit {
        name = "noisy-state-lib-src";
        url = https://github.com/HWQuantum/noisy-state-lib;
        ref = "refs/heads/master";
        rev = "<REPLACE WITH NEWEST REVISION>";
      };
      propagatedBuildInputs = with pkgs.python3Packages; [numpy];
    };
in
with pkgs;
mkShell {
  name = "import_packages";
  buildInputs = [
    (python3.withPackages (ps: with ps; [mode_sorting numpy]))
  ];
}
```

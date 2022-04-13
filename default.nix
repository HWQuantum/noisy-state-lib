{lib, buildPythonPackage, pytest, numpy, setuptools, yapf, matplotlib, numba}:
buildPythonPackage rec {
  pname = "noisy-state-lib";
  version = "0.0.1";
  src = "./";
  nativeBuildInputs = [setuptools];
  propagatedBuildInputs = [numpy numba matplotlib];
  checkInputs = [pytest];
}

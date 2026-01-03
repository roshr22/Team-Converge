"""Setup script for deepfake-patch-audit."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="deepfake-patch-audit",
    version="0.1.0",
    description="Deepfake detection using patch-level teacher-student distillation with quantization auditing",
    author="Team Converge",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=required,
    entry_points={
        "console_scripts": [
            "deepfake-inference=inference.pipeline:main",
            "deepfake-eval=scripts.run_eval:main",
        ],
    },
    include_package_data=True,
)

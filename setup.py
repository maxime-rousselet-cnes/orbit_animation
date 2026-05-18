"""
To install via pip install -e . in the root of the repository. This will make the ORBIT_ANIMATION
package available in the current environment.
"""

from setuptools import find_packages, setup

setup(
    name="orbit_animation",
    packages=find_packages(),
    version="0.0.1",
    description="Uses manim library to display animations explaining orbit determination.",
    author="Maxime Rousselet",
)

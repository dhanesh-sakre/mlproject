from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path:str)->List[str]:
    '''
    this function returns a list of requirements
    '''
    HYPHEN_DOT_E = '-e .'
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)

    return requirements


setup(
    name="MachineLearningProject",
      version="0.0.1",
      author="Dhanesh", 
      author_email="dhaneshsakre@gmail.com",
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')
      
)
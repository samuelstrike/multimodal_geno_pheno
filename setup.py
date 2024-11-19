from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    Read the requirements file and return the list of requirements.
    '''

    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readline()
        [req.replace("\n"," ")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name='MMDL',
    packages=find_packages(),
    version='0.0.1',
    author='Samuel Pradhan',
    author_email='samuel.pradhan22@gmail.com',
    install_requires=get_requirements('requirements.txt')
)
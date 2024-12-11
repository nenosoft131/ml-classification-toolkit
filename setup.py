from setuptools import find_packages, setup
from typing import List

HYPEN_E = '-e .'
def get_requirements(fpath:str)->List[str]:
    
    requirements=[]
    with open(fpath) as fobj:
        requirements =fobj.readline()
        requirements= [req.replace("\n","") for req in requirements]
        
        if HYPEN_E in requirements:
            requirements.remove(HYPEN_E)
    
    return requirements


setup(
    name = 'MlPro',
    version= '0.1',
    author= 'Muhammad',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')
)

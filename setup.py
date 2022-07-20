from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="rtd",
        author="Ilya Trofimov",
        packages=find_packages(),
        version = '1.0',
        install_requires=['numpy', 'scipy'])

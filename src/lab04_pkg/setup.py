from setuptools import find_packages, setup

package_name = 'lab04_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='francesca.fornasier@icloud.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['task0=lab04_pkg.task0:main',
                            'task1=lab04_pkg.task1:main',
                            'task2=lab04_pkg.task2:main',
                            'task3=lab04_pkg.task3:main',
        ],
    },
)

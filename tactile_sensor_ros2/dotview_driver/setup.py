from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dotview_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name,'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shawn_shi',
    maintainer_email='shi_xh@zju.edu.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_stream = dotview_driver.sensor_stream:main',
            'feature_extraction = dotview_driver.feature_extraction:main',
            'mysocket = dotview_driver.mysocket:main',
            'save_video = dotview_driver.save_video:main',
            'depth_measure = dotview_driver.depth_measure:main',
        ],
    },
)

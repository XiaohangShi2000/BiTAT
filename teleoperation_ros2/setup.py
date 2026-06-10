from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tele_vtrm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shawn shi',
    maintainer_email='shi_xh@zju.edu.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control = tele_vtrm.control:main',
            'record_right = tele_vtrm.record_right:main',
            'record_bi = tele_vtrm.record_bi:main',
            'eval_right_control = tele_vtrm.eval_right_control:main',
            'eval_right_infer = tele_vtrm.eval_right_infer:main',
            'eval_bi_control = tele_vtrm.eval_bi_control:main',
            'eval_bi_infer = tele_vtrm.eval_bi_infer:main',
        ],
    },
)

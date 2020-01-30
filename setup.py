from setuptools import setup, find_packages
from duffingtools import __version__ as current_version

# NOTES for updating this file:
# 1) for version update in the duffingtools.__init__
# 2) update the following comment_on_changes
comment_on_changes = 'test_01'

setup(
    name='test',
    version=current_version,
    packages=find_packages(),
    # package_data={'pylabcontrol': ['gui/ui_files/*ui']},
    url='https://github.com/MarcBala/data_analysis_tools',
    license='BSD-2-Clause',
    author='Marc Torrent',
    author_email='test',
    description='Duffing oscillator theory and fitting',
    keywords='duffing oscillator, data analysis',
    long_description=comment_on_changes,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: 2-Clause BSD License',
        'Development Status :: 4 - Beta',
        'Environment :: Linux (Ubuntu)',
        ],
    install_requires=[
        'matplotlib',
        'pandas',
        'numpy',
        'scipy',
        'lmfit',
        'uncertainties',
    ]
    # test_suite='nose.collector',
    # tests_require=['nose'],
    # python_requires='>=3.6',
    # entry_points={
    #     'console_scripts': ['pylabcontrol = pylabcontrol.gui.launch_gui:launch_gui']
    # }
)

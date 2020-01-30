from setuptools import setup, find_packages
from data_analysis_tools import __version__ as current_version

# NOTES for updating this file:
# 1) for version update in the data_analysis_tools.__init__
# 2) update the following comment_on_changes
comment_on_changes = 'changes version to 0.0a2'

setup(
    name='data_analysis_tools',
    version=current_version,
    packages=find_packages(),
    # package_data={'pylabcontrol': ['gui/ui_files/*ui']},
    url='https://github.com/MarcBala/data_analysis_tools.git',
    license='BSD-2-Clause',
    author='Marc Torrent',
    author_email='marccuairan@gmail.com',
    description='asdasdasd',
    keywords='data analysis',
    long_description=comment_on_changes,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: 2-Clause BSD License',
        'Development Status :: 4 - Beta',
        'Environment :: Linux (Ubuntu)',
        ],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
    ]
    # test_suite='nose.collector',
    # tests_require=['nose'],
    # python_requires='>=3.6',
    # entry_points={
    #     'console_scripts': ['pylabcontrol = pylabcontrol.gui.launch_gui:launch_gui']
    # }
)

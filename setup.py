from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_install_requires():
    install_requires = [
        "tqdm",
        "opencv_python",  # for PyInstaller
        "numpy",
        "pycocotools",
    ]
    return install_requires


setup(
    # 取名不能够用_会自动变-
    name="ccdt",
    version="0.0.1",
    packages=find_packages(exclude=["data"]),
    install_requires=get_install_requires(),
    author="zhanyong",
    author_email="zhan.yong@chipeak.com",
    description="AI数据转换工具箱",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/540717421/chipeak_cv_data_tool",
    project_urls={
        "Bug Tracker": "https://github.com/540717421/chipeak_cv_data_tool/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    # package_data={"cpdt": ["icons/*", "config/*.yaml"]},
    entry_points={
        "console_scripts": [
            "ccdt=ccdt.dataset.__main__:main",
            # "labelme=labelme.__main__:main",
            # "labelme_draw_json=labelme.cli.draw_json:main",
            # "labelme_draw_label_png=labelme.cli.draw_label_png:main",
            # "labelme_json_to_dataset=labelme.cli.json_to_dataset:main",
            # "labelme_on_docker=labelme.cli.on_docker:main",
        ],
    },
    # package_dir={"": "src"},
    # packages=setuptools.find_packages(where="src"),
    # packages=find_packages(exclude=('configs', 'tools', 'demo')),
    # package_dir={"chipeak_data_tool": "chipeak_data_tool"},
    # packages=setuptools.find_packages(include=['chipeak_data_tool.*']),
    # python_requires=">=3.7",
)

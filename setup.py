from setuptools import Extension, setup
from pathlib import Path
import numpy as np
import cmake_build_extension
import importlib
import platform
import subprocess
import os
import shutil
from setuptools.command.build_ext import build_ext


class MyBuildExtension(build_ext):
    def initialize_options(self):

        # Initialize base class
        build_ext.initialize_options(self)

        # Initialize the '--define' custom option, overriding the pre-existing one.
        # Originally, it was aimed to pass C preprocessor definitions, but instead we
        # use it to pass custom configuration options to CMake.
        self.define = None

        # Initialize the '--component' custom option.
        # It overrides the content of the cmake_component option of CMakeExtension.
        self.component = None

        # Initialize the 'no-cmake-extension' custom option.
        # It allows disabling one or more CMakeExtension from the command line.
        self.no_cmake_extension = None

    @staticmethod
    def extend_cmake_prefix_path(path: str) -> None:

        abs_path = Path(path).absolute()

        if not abs_path.exists():
            raise ValueError(f"Path {abs_path} does not exist")

        if "CMAKE_PREFIX_PATH" in os.environ:
            os.environ[
                "CMAKE_PREFIX_PATH"
            ] = f"{str(path)}:{os.environ['CMAKE_PREFIX_PATH']}"
        else:
            os.environ["CMAKE_PREFIX_PATH"] = str(path)

    def finalize_options(self):

        # Parse the custom CMake options and store them in a new attribute
        defines = [] if self.define is None else self.define.split(";")
        self.cmake_defines = [f"-D{define}" for define in defines]

        # Parse the disabled CMakeExtension modules and store them in a new attribute
        self.no_cmake_extensions = (
            []
            if self.no_cmake_extension is None
            else self.no_cmake_extension.split(";")
        )

        # Call base class
        build_ext.finalize_options(self)

    def run(self) -> None:
        """
        Process all the registered extensions executing only the CMakeExtension objects.
        """

        # Filter the CMakeExtension objects
        cmake_extensions = [e for e in self.extensions if isinstance(e, cmake_build_extension.CMakeExtension)]

        if len(cmake_extensions) == 0:
            raise ValueError("No CMakeExtension objects found")

        # Check that CMake is installed
        if shutil.which("cmake") is None:
            raise RuntimeError("Required command 'cmake' not found")

        # Check that Ninja is installed
        if shutil.which("ninja") is None:
            raise RuntimeError("Required command 'ninja' not found")

        for ext in cmake_extensions:

            # Disable the extension if specified in the command line
            if (
                ext.name in self.no_cmake_extensions
                or "all" in self.no_cmake_extensions
            ):
                continue

            # Disable all extensions if this env variable is present
            disabled_set = {"0", "false", "off", "no"}
            env_var_name = "CMAKE_BUILD_EXTENSION_ENABLED"
            if (
                env_var_name in os.environ
                and os.environ[env_var_name].lower() in disabled_set
            ):
                continue

            self.my_build_extension(ext)
        self.extensions = [e for e in self.extensions if not isinstance(e, cmake_build_extension.CMakeExtension)]
        build_ext.run(self)

    # don't use Ninja
    def my_build_extension(self, ext: cmake_build_extension.CMakeExtension) -> None:
        """
        Build a CMakeExtension object.

        Args:
            ext: The CMakeExtension object to build.
        """

        # if self.inplace and ext.disable_editable:
        #     print(f"Editable install recognized. Extension '{ext.name}' disabled.")
        #     return

        # Export CMAKE_PREFIX_PATH of all the dependencies
        for pkg in ext.cmake_depends_on:

            try:
                importlib.import_module(pkg)
            except ImportError:
                raise ValueError(f"Failed to import '{pkg}'")

            init = importlib.util.find_spec(pkg).origin
            cmake_build_extension.BuildExtension.extend_cmake_prefix_path(path=str(Path(init).parent))

        cmake_install_prefix = ext.install_prefix

        # CMake configure arguments
        configure_args = [
            # '-GNinja',
            f"-DCMAKE_BUILD_TYPE={ext.cmake_build_type}",
            f"-DCMAKE_INSTALL_PREFIX:PATH={cmake_install_prefix}",
            # Fix #26: https://github.com/diegoferigo/cmake-build-extension/issues/26
            # f"-DCMAKE_MAKE_PROGRAM={shutil.which('ninja')}",
        ]

        # Extend the configure arguments with those passed from the extension
        configure_args += ext.cmake_configure_options

        # CMake build arguments
        build_args = ["--config", ext.cmake_build_type]

        if platform.system() == "Windows":

            configure_args += []

        elif platform.system() in {"Linux", "Darwin"}:

            configure_args += ['-DCMAKE_C_FLAGS="-fPIC"']

        else:
            raise RuntimeError(f"Unsupported '{platform.system()}' platform")

        # Parse the optional CMake options. They can be passed as:
        #
        # python setup.py build_ext -D"BAR=Foo;VAR=TRUE"
        # python setup.py bdist_wheel build_ext -D"BAR=Foo;VAR=TRUE"
        # python setup.py install build_ext -D"BAR=Foo;VAR=TRUE"
        # python setup.py install -e build_ext -D"BAR=Foo;VAR=TRUE"
        # pip install --global-option="build_ext" --global-option="-DBAR=Foo;VAR=TRUE" .
        #
        configure_args += self.cmake_defines

        # Get the absolute path to the build folder
        build_folder = str(Path(".").absolute() / f"{self.build_temp}_{ext.name}")

        # Make sure that the build folder exists
        Path(build_folder).mkdir(exist_ok=True, parents=True)

        # 1. Compose CMake configure command
        configure_command = [
                                "cmake",
                                "-S",
                                ext.source_dir,
                                "-B",
                                build_folder,
                            ] + configure_args

        # 2. Compose CMake build command
        build_command = ["cmake", "--build", build_folder] + build_args

        # 3. Compose CMake install command
        install_command = ["cmake", "--install", build_folder]

        # If the cmake_component option of the CMakeExtension is used, install just
        # the specified component.
        if self.component is None and ext.cmake_component is not None:
            install_command.extend(["--component", ext.cmake_component])

        # Instead, if the `--component` command line option is used, install just
        # the specified component. This has higher priority than what specified in
        # the CMakeExtension.
        if self.component is not None:
            install_command.extend(["--component", self.component])

        print("")
        print("==> Configuring:")
        print(f"$ {' '.join(configure_command)}")
        print("")
        print("==> Building:")
        print(f"$ {' '.join(build_command)}")
        print("")
        print("==> Installing:")
        print(f"$ {' '.join(install_command)}")
        print("")

        # Call CMake
        subprocess.check_call(configure_command)
        subprocess.check_call(build_command)
        subprocess.check_call(install_command)

        # Write content to the top-level __init__.py
        if ext.write_top_level_init is not None:
            with open(file=cmake_install_prefix / "__init__.py", mode="w") as f:
                f.write(ext.write_top_level_init)

        # Write content to the bin/__main__.py magic file to expose binaries
        if len(ext.expose_binaries) > 0:
            bin_dirs = {str(Path(d).parents[0]) for d in ext.expose_binaries}

            import inspect

            main_py = inspect.cleandoc(
                f"""
                from pathlib import Path
                import subprocess
                import sys

                def main():

                    binary_name = Path(sys.argv[0]).name
                    prefix = Path(__file__).parent.parent
                    bin_dirs = {str(bin_dirs)}

                    binary_path = ""

                    for dir in bin_dirs:
                        path = prefix / Path(dir) / binary_name
                        if path.is_file():
                            binary_path = str(path)
                            break

                        path = Path(str(path) + ".exe")
                        if path.is_file():
                            binary_path = str(path)
                            break

                    if not Path(binary_path).is_file():
                        name = binary_path if binary_path != "" else binary_name
                        raise RuntimeError(f"Failed to find binary: {{ name }}")

                    sys.argv[0] = binary_path

                    result = subprocess.run(args=sys.argv, capture_output=False)
                    exit(result.returncode)

                if __name__ == "__main__" and len(sys.argv) > 1:
                    sys.argv = sys.argv[1:]
                    main()"""
            )

            bin_folder = cmake_install_prefix / "bin"
            Path(bin_folder).mkdir(exist_ok=True, parents=True)
            with open(file=bin_folder / "__main__.py", mode="w") as f:
                f.write(main_py)


extensions = [
    Extension(
        'v3dpy.loaders.pbd',
        ['v3dpy/loaders/pbd.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
    Extension(
        'v3dpy.loaders.raw',
        ['v3dpy/loaders/raw.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
    Extension(
        'v3dpy.terafly.tiff_manage',
        ['v3dpy/terafly/tiff_manage.pyx'],
        include_dirs=['3rdparty/libtiff/include', np.get_include()],
        library_dirs=['3rdparty/libtiff/lib'],
        libraries=['tiff'],
        language='c++'
    ),
    Extension(
        'v3dpy.terafly.format_managers',
        ['v3dpy/terafly/format_managers.pyx'],
        include_dirs=[np.get_include()],
        language='c++'
    ),
    Extension(
        'v3dpy.terafly.volume_managers',
        ['v3dpy/terafly/volume_managers.pyx'],
        include_dirs=[np.get_include()],
        language='c++',
    ),
]

setup(
    ext_modules=[
        cmake_build_extension.CMakeExtension(
        'tiff',
        '3rdparty/libtiff',
        source_dir=str(Path('3rdparty/libtiff').absolute()),
        cmake_configure_options=[]
    )] + extensions,
    cmdclass=dict(
        # Enable the CMakeExtension entries defined above
        build_ext=MyBuildExtension,
        # If the setup.py or setup.cfg are in a subfolder wrt the main CMakeLists.txt,
        # you can use the following custom command to create the source distribution.
        # sdist=cmake_build_extension.GitSdistFolder
    ),
)

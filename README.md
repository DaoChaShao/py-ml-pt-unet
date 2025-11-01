<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
The **Cambridge-driving Labeled Video Database (CamVid)** is one of the first automotive datasets to provide
object-class semantic labels along with metadata for computer vision applications. This dataset addresses the need for
more complex scene understanding beyond simple object detection, focusing on pixel-level semantic segmentation tasks.

**DATA DESCRIPTION**
---

**First** publicly available dataset with semantic labeling for automotive applications
**Pixel-level** annotations for 32 semantic classes
**High-resolution** 960×720 video sequences
**Temporal correspondence** between frames with metadata

**Semantic Classes (32 classes)**
---

1. **Sky** - Atmospheric regions
2. **Building** - Man-made structures
3. **Pole** - Utility poles, traffic poles
4. **Road** - Driveable road surfaces
5. **Pavement** - Sidewalks and pedestrian paths
6. **Tree** - Vegetation and trees
7. **SignSymbol** - Traffic signs and symbols
8. **Fence** - Barriers and fences
9. **Car** - Vehicles and cars
10. **Pedestrian** - People walking
11. **Bicyclist** - Cyclists and bicycles
12. **Unlabeled** - Unclassified pixels

Plus 20 additional specialized classes including: Bridge, Wall, GuardRail, etc.

**PRIVACY NOTICE**
---

This application is designed to process the data you provide to generate customized suggestions and results. Your
privacy is paramount.

**We do not collect, store, or transmit your personal information or data.** All processing occurs locally on your
device (in your browser or runtime environment), and **no data is ever sent to an external server or third party.**

- **Local Processing:** Your data never leaves your device. The entire analysis and generation process happens locally.
- **No Data Retention:** Since no data is transmitted, none is stored on any server. Closing the application typically
  clears any temporary local data.
- **Transparency:** The entire codebase is open source. You are encouraged to review the [code](./) to verify how your
  data is handled.

In summary, you maintain full control and ownership of your data at all times.

**LICENCE**
---
This project is open source and available under the **[BSD-3-Clause Licence](LICENCE)**.

In simple terms, this is a very permissive licence that allows you to freely use this code for almost any purpose,
including in proprietary projects, as long as you include the original copyright and licence notice.

Feel free to fork, modify, and build upon this work! We simply ask that you give credit where credit is due.

**ENVIRONMENT SETUP**
---
This project uses **Python 3.12** and [uv](https://docs.astral.sh/uv/) for fast dependency management and virtual
environment handling. The required Python version is automatically detected from the [.python-version](.python-version)
file.

1. **Installing uv**:  
   If you don't have `uv` installed, you can install it using the following command:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # This installation method works on macOS and Linux.
    ```
   Alternatively, you can install it by running the following PowerShell command:
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # This installation method works on Windows.
    ```

   **💡 Recommended**: For the best experience, install `uv` as a standalone tool. Avoid installing it within `pip` or
   `conda` environments to prevent potential dependency conflicts.

2. **Adding Dependencies**:

- To add a main (production) dependency:
    ```bash
    uv add <package_name>
    # This automatically updates pyproject.toml and installs the package
    ```
- To add a development dependency:
    ```bash
    uv add <package_name> --group dev
    # Example: uv add ruff --group dev
    # This adds the package to the [project.optional-dependencies.dev] section automatically
    ```
- To add other types of optional dependencies (e.g., test, docs):
    ```bash
    uv add <package_name> --group test
    uv add <package_name> --group docs
    ```
- To import dependencies from a `requirements.txt` file:
    ```bash
    uv add -r requirements.txt
    # This reads packages from requirements.txt and adds them to pyproject.toml
    ```

3. Removing Dependencies

- To remove a main (production) dependency:
    ```bash
    uv remove <package_name>
    # This automatically updates pyproject.toml and removes the package
    ```
- To remove a development dependency:
    ```bash
    uv remove <package_name> --group dev
    # Example: uv remove ruff --group dev
    # This removes the package from the [project.optional-dependencies.dev] section
    ```
- To remove other types of optional dependencies:
    ```bash
    uv remove <package_name> --group test
    uv remove <package_name> --group docs
    ```

4. **Managing the Environment**

- After using add/remove commands, sync the environment:
    ```bash
    uv sync
    ```

**CHANGELOG**
---
This project uses [git-changelog](https://github.com/pawamoy/git-changelog) to automatically generate and maintain a
changelog based on [Conventional Commits](https://www.conventionalcommits.org/).

1. **Installation**
   ```bash
   pip install git-changelog
   # or use uv to add it as a development dependency
   uv add git-changelog --group dev
   ```
2. **Verify Installation**
   ```bash
   pip show git-changelog
   # or check the version specifically
   pip show git-changelog | grep Version
   ```
3. **Configuration**
   Ensure you have a properly configured `pyproject.toml` file at the project root. The configuration should specify
   Conventional Commits as the changelog style. Here is an example configuration:
   ```toml
   [tool.git-changelog]
   version = "0.1.0"
   style = "conventional-commits"
   output = "CHANGELOG.md"
   ```
4. **Generate Changelog**
   ```bash
   git-changelog --output CHANGELOG.md
   # Or use uv to run it if installed as a dev dependency
   uv run git-changelog --output CHANGELOG.md
   ```
   This command creates or updates the `CHANGELOG.md` file with all changes based on your git history.
5. **Push Changes**
   ```bash
   git push origin main
   ```
   Alternatively, use your IDE's Git interface (e.g., `Git → Push` in many editors).
6. **Note**:

- The changelog is automatically generated from your commit messages following the Conventional Commits specification.
- Run the generation command whenever you want to update the changelog, typically before a release or after significant
  changes.

# LHAI

<p align="center">
 <img alt="" src="/docs/assets/images/logo1.png" width=500 align="center">
</p>

这是LHAI的用户手册。LHAI项目旨在运用AI技术进行LHAASO(The Large High Altitude Air Shower Observatory)的数据处理。

<p align="center"><a href="https://github.com/FreeTubeApp/FreeTube/releases">Download FreeTube</a></p>
<p align="center">
  <a href="https://github.com/FreeTubeApp/FreeTube/actions/workflows/build.yml">
    <img alt='Build status' src="https://github.com/FreeTubeApp/FreeTube/actions/workflows/build.yml/badge.svg?branch=development" />
  </a>
  <a href="https://hosted.weblate.org/engage/free-tube/">
    <img src="https://hosted.weblate.org/widgets/free-tube/-/svg-badge.svg" alt="Translation status" />
  </a>
</p>

<hr>
<p align="center"><a href="#1-README">README</a> &bull; <a href="#2-How-Does-This-Manual-Work">How Does This Manual Work</a> &bull; <a href="#3-Structure-of-This-Manual">Structure of This Manual</a> &bull; <a href="#4-Becoming-a-Collaborator">Becoming a Collaborator</a> &bull; <a href="#5-Resources">Resources</a></p>
<hr>

> [!NOTE]  
> This manual is actively maintained and updated by contributors. While we strive to keep the content accurate and comprehensive, there may still be areas for improvement or updates pending.  
> 
> If you spot any issues or have suggestions for improvement, please submit a [GitHub issue](https://github.com/boobyuuuu/LHAI/issues/new/choose) to help us track and address it. Make sure to check [existing issues](https://github.com/boobyuuuu/LHAI/issues) first to avoid duplicates!


## 1 README

- [查看中文版](README_zh.md)
 
- [English version](README_en.md)

## 2 How Does This Manual Work

This user manual is based on **MkDocs** for webpage construction, utilizes **Git** for version control and collaborative management, and is published online through **GitHub Pages**.

### **MkDocs**

**MkDocs** is a documentation tool designed for quickly generating static websites. It supports writing in simple Markdown syntax and produces clean, structured HTML websites.

- **Installation and Usage**:
  1. Install MkDocs:  
     ```bash
     pip install mkdocs
     ```
  2. Create a project:  
     ```bash
     mkdocs new my-project
     cd my-project
     ```
  3. Run a local preview:  
     ```bash
     mkdocs serve
     ```
     Visit `http://127.0.0.1:8000` to preview the documentation.

  4. Build a static website:  
     ```bash
     mkdocs build
     ```
     The generated files will be stored in the `site` directory.

- **Themes and Plugins**:  
  Use the default MkDocs theme (`mkdocs`) or extended themes like `mkdocs-material`. Additional functionality can be added through plugins configured in the `mkdocs.yml` file.

### **Git**

**Git** is a distributed version control system for managing documentation and code versions while enabling team collaboration.

- **Initialization and Cloning**:
  1. Initialize a project:  
     ```bash
     git init
     ```
  2. Clone a remote repository:  
     ```bash
     git clone <repository-url>
     ```

- **Common Commands**:
  - Commit changes:  
    ```bash
    git add .
    git commit -m "Update documentation"
    ```
  - Push to the remote repository:  
    ```bash
    git push origin main
    ```

- **Branch Management**: Use branches to avoid conflicts during multi-user collaboration:  
  ```bash
  git checkout -b new-feature
  ```

### **GitHub Pages**

**GitHub Pages** is a hosting service for static websites, allowing you to deploy MkDocs-generated sites directly from GitHub repositories.

- **Deployment Steps**:
  1. Configure the deployment branch in `mkdocs.yml`:  
     ```yaml
     site_name: My Documentation
     site_dir: site
     ```
  2. Deploy using `gh-deploy`:  
     ```bash
     mkdocs gh-deploy
     ```
  3. Once deployed, the documentation will be available at `https://<username>.github.io/<repository>`.

- **Automated Deployment**:  
  Use GitHub Actions to automate the build and deployment process. Example workflow:
  ```yaml
  name: Deploy MkDocs
  on:
    push:
      branches:
        - main
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v2
        - name: Set up Python
          uses: actions/setup-python@v2
          with:
            python-version: 3.x
        - name: Install dependencies
          run: pip install mkdocs mkdocs-material
        - name: Build and deploy
          run: mkdocs gh-deploy --force
  ```

## 3 Structure of This Manual

**Homepage**: Contains a directory and visual representation of results (Results Section).

**Contents**:
- **Data Section**
- **Technical Section**
- **Validation Section**
- **Code Structure Explanation**
- **Other Technical Documents** (Git, Markdown, MkDocs, Server)

## 4 Becoming a Collaborator

### **Prerequisites**:
- Familiarity with Git, a VPN, and Python 3.10 or above.

### **Steps**:
1. **Create a Local Repository and Link It to GitHub**:
   - Create an empty folder for storing local documentation files.
   - Clone the GitHub repository:  
     ```bash
     git clone https://github.com/boobyuuuu/LHAI.git
     ```

   Note: Errors may occur due to initial environment settings. Address these errors as they arise.

2. **Set Up MkDocs Environment**:
   - Install the necessary dependencies:  
     ```bash
     pip install mkdocs
     pip install mkdocs-material
     pip install pymdown-extensions
     ```
   - Run the local preview:  
     ```bash
     mkdocs serve
     ```

3. **Edit the Manual**:
   - Modify as needed.
   - The `mkdocs.yml` file contains all configuration and structure information.
   - Refer to the MkDocs tutorial:  
     [MkDocs-Like-Code](https://mkdocs-like-code.readthedocs.io/zh-cn/latest/)

4. **Upload Changes to GitHub and GitHub Pages**:
   - Push to GitHub:  
     ```bash
     git add .
     git commit -m "English log for modifications"
     git push
     ```
   - Deploy to GitHub Pages:  
     ```bash
     mkdocs gh-deploy
     ```

5. **Daily Workflow**:
   - Pull the latest version from GitHub:  
     ```bash
     git pull
     ```
   - Make local changes and push them back to GitHub.
   - Deploy updates to GitHub Pages.

**Follow this workflow strictly.**

## 5 未来的研究目标与问题

### 1.在ihep老师完成程序开发之前

a.模拟intensity map（内禀（intrinsic）图像）而非用一个更小的PSF卷积的图像

b.更多类型的源 （生成更不规则的源）

c.gamma背景仍然使用Fermi LAT diffuse gamma； 宇宙线背景的分布调研。

> [!TIPS]
>
> 背景分为宇宙线背景（被错误分类为gamma射线的宇宙线），和伽马射线背景（弥散的辐射）。目前我们仅添加了弥散背景，使用Fermi LAT发布的伽马射线弥散背景模型。后续需要明确宇宙线背景的分布规律并添加到训练数据中。（宇宙线背景通常用多个背源窗口作平均得到，因此可能不服从Poisson分布。）

d.不同机器学习模型, 尝试论文中的模型、前沿的超分辨技术 （cly）

e.了解LHAASO的项目以及数据处理相关内容（hxy）

f.完善代码注释，适当重写，用户手册撰写 （gxw, lzh）

g.改进模型的损失函数等 （种类、比例……）

### 2.在拿到程序之后

a.足够多的样本生成，前期阶段性成果

b.对不同PSF进行泛化（模态融合？ ）


## 6 Resources

### **About LHAI**
- [Poster: The Second LHAASO Collaboration Conference in 2024](docs/resources/final_poster.pptx)
- [Xingwei's PPT about LHAI](docs/resources/241019.pdf)
- [Zihang's PPT on the Conference](docs/resources/POS_64.PPTX)

### **About LHAASO**
- [Traditional Methods](docs/resources/利用LHAASO-KM2A...研究及ED-PMT批量测试_于艳红.pdf) 
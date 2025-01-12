# LHAI项目代码：docs文件夹

`docs` 文件夹用于存放LHAI的用户手册，本用户手册以 `mkdocs` 格式书写。

## Folder Structure  

```
├── README.md                  <- Top-level README for developers using this project
├── mkdocs.yml                 <- Main configuration file for MkDocs
├── requirements.txt           <- Python dependencies for the MkDocs project
├── Content                    <- Folder containing the documentation content
│   ├── data                   <- Section for data-related documentation
│   │   ├── index.md           <- Overview page for the Data Section
│   │
│   ├── network                <- Section for network-related documentation
│   │   ├── index.md           <- Overview page for the Network Section
│   │
│   ├── eval                   <- Section for evaluation-related documentation
│   │   ├── index.md           <- Overview page for the Eval Section
│   │
│   ├── codes                  <- Section for code-related documentation
│   │   ├── index.md           <- Overview page for the Codes Section
│   │   ├── data.md            <- Documentation for data usage in codes
│   │   ├── docs.md            <- Documentation for code structure and details
│   │
│   ├── others                 <- Section for miscellaneous content
│   │   ├── index.md           <- Overview page for the Others Section
│   │   ├── 1.md               <- Documentation guidance or additional content
│
├── docs                       <- Default MkDocs output folder for site generation
│
├── css                        <- Custom CSS for styling the MkDocs project
│   ├── base.css               <- Base styling for the entire site
│   ├── theme-overrides.css    <- Customizations to override the default MkDocs theme
│   ├── typography.css         <- Specific typography adjustments for better readability
│   └── responsive.css         <- Styles for ensuring responsive design across devices
│
└── assets                     <- Static assets like images or fonts
    ├── images                 <- Images used throughout the documentation
    │   ├── logo.png           <- Site logo
    │   ├── favicon.ico        <- Favicon for the site
    │
    ├── fonts                  <- Custom fonts for the site, if needed
```

## 这个手册如何运行？

本用户手册基于 **MkDocs** 进行网页构建，通过 **Git** 管理代码版本控制和多用户协作，借助 **GitHub Pages** 实现在线发布与分享。

### MkDocs

**MkDocs** 是一个用于快速生成静态网站的文档工具，支持简单的 Markdown 格式书写文档，并生成结构清晰的 HTML 网站。

- 安装与使用：

  1. 安装 MkDocs：  

     ```bash
     pip install mkdocs
     ```

  2. 创建项目：  

     ```bash
     mkdocs new my-project
     cd my-project
     ```

  3. 本地运行预览：  

     ```bash
     mkdocs serve
     ```

     访问 `http://127.0.0.1:8000` 查看文档效果。

  4. 构建静态网站：  

     ```bash
     mkdocs build
     ```

     所有生成的文件将存储在 `site` 文件夹中。

- 主题与插件：  

  使用 MkDocs 默认主题 `mkdocs` 或扩展主题如 `mkdocs-material`。可通过 `mkdocs.yml` 配置文件添加插件实现功能扩展。

### Git

Git 是分布式版本控制系统，用于管理文档与代码版本，同时支持团队协作。

- 初始化与克隆：
  1. 初始化项目：  

     ```bash
     git init
     ```

  2. 克隆远程仓库：  

     ```bash
     git clone <repository-url>
     ```

- 常用命令：

  - 提交更改：  

    ```bash
    git add .
    git commit -m "Update documentation"
    ```

  - 推送到远程仓库：  

    ```bash
    git push origin main
    ```

- 分支管理：多用户协作时，可使用分支功能避免冲突：  

  ```bash
  git checkout -b new-feature
  ```

### GitHub Pages

**GitHub Pages** 是一个托管静态网页的服务，可以直接从 GitHub 仓库中托管 MkDocs 生成的网站。

- 发布步骤：

  1. 在 `mkdocs.yml` 中配置部署分支：  

     ```yaml
     site_name: My Documentation
     site_dir: site
     ```

  2. 使用 `gh-deploy` 部署： 

     ```bash
     mkdocs gh-deploy
     ```

  3. 部署完成后，文档将发布到 `https://<username>.github.io/<repository>`。

- 自动化部署：  

  借助 GitHub Actions，可自动触发构建与部署流程。例如：

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

## 本手册结构

首页：目录 + 结果直观表示（结果篇）

内容：

- 数据篇

- 技术篇

- 验证篇

- 代码结构详解

- 其他技术文档（GIT,MARKOWN,MKDOCS,SERVER）

## 成为共同创作者

前置条件：git、梯子、python3.10以上

1. 创建本地仓库，与git仓库连接

    （1）创建一个空白文件夹，用于存储文档本地文件

    （2）克隆github项目文档：

    ```
    git clone https://github.com/boobyuuuu/LHAI.git
    ```

    这一步很大概率会报错，是初始环境设置的问题；根据具体报错具体解决

2. 拿到项目用户手册文件 LHAI 后，可以进行查看：

    （1）用户手册基于mkdocs构建。首先是mkdocs的环境：

    ```
    pip install mkdocs

    pip install mkdocs-material

    pip install pymdown-extensions
    ```

    以上三个是主要的环境，还有一些插件，根据报错差哪些就安装哪些，知道可以运行：

    ```
    mkdocs serve
    ```

3. 修改用户手册

    （1）根据自己的需要进行修改

    （2）mkdocs.yml 文件存储了所有的配置信息、网页结构信息

    （3）具体mkdocs教程查看：

    https://mkdocs-like-code.readthedocs.io/zh-cn/latest/

4. 确认自己修改没问题后，上传到 github，上传到 github-pages

    （1）上传github连招：

    ```
    git add .

    git commit -m "英文填写修改日志"

    git push
    ```

    （2）上传github-pages：

    ```
    mkdocs gh-deploy
    ```

5. 平常工作流程

    由于是多人协同，共同管理形式，需要每个人默契配合，并形成严谨的修改规则。工作流程如下：

    （1）自己进行了一些工作，在自己本地写了新的内容

    （2）同步github的版本：

    ```
    git pull
    ```
    这一步会将github上最新的版本拉取到自己本地，但只会对修改过的部分进行更改，不会把你创建的新的本地文档覆盖消除掉

    （3）进行github上传

    （4）进行github-pages上传

    **请遵守这样的工作流程**

<p align='right'>by Zihang Liu</p>
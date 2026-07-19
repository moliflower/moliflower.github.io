// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-首页",
    title: "首页",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-博客",
          title: "博客",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-项目",
          title: "项目",
          description: "我参与的研究、工程与开源项目。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-简历",
          title: "简历",
          description: "Yuhan Deng 的个人简历。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-位置编码-从-vit-patch-到-llm-rope",
        
          title: "位置编码：从 ViT Patch 到 LLM RoPE",
        
        description: "从注意力的置换等变性出发，理解绝对位置、相对位置、ViT 的 patch 位置嵌入与大语言模型中的旋转位置编码 RoPE。",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/%E4%BB%8E-vit-patch-%E5%88%B0-llm-rope-%E7%90%86%E8%A7%A3%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/";
          
        },
      },{id: "post-理解大模型中的升维与降维-从张量形状到-ffn",
        
          title: "理解大模型中的升维与降维：从张量形状到 FFN",
        
        description: "从非线性表达、Key-Value 记忆、特征叠加与工程解耦四个视角，理解 Transformer FFN 为什么先升维再降维。",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/%E5%AF%B9%E4%BA%8E%E6%A8%A1%E5%9E%8B%E4%B8%AD%E5%8D%87%E7%BB%B4%E5%92%8C%E9%99%8D%E7%BB%B4%E7%9A%84%E7%90%86%E8%A7%A3/";
          
        },
      },{id: "post-实验室服务器-ssh-入门-从-remote-ssh-到-github-ssh-key",
        
          title: "实验室服务器 SSH 入门：从 Remote SSH 到 GitHub SSH Key",
        
        description: "记录第一次使用实验室服务器时，从 VS Code Remote SSH 登录、免密认证，到解决 GitHub push 反复认证问题的过程。",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/%E8%80%81%E5%B8%88%E5%8F%AA%E7%BB%99%E6%88%91%E4%B8%80%E4%B8%AAip-%E7%A7%91%E7%A0%94%E5%B0%8F%E7%99%BD%E7%9A%84ssh%E5%85%A5%E9%97%A8/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-torchmorph-一个轻量级-cuda-加速库-用于在-pytorch-中实现批量并行的形态学变换",
          title: 'TorchMorph 一个轻量级 CUDA 加速库，用于在 PyTorch 中实现批量并行的形态学变换。',
          description: "",
          section: "News",},{id: "projects-torchmorph",
          title: 'TorchMorph',
          description: "面向 PyTorch 的 CUDA 加速批量并行形态学变换库。",
          section: "Projects",handler: () => {
              window.location.href = "/projects/torchmorph/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/example_pdf.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];

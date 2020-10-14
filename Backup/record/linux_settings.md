### 安装Nvidia显卡驱动

sudo apt remove --purge nvidia*                      # 卸载已有的nvidia显卡驱动（如果已安装的话）
sudo add-apt-repository ppa:graphics-drivers/ppa     # 添加ppa源
sudo apt update                                      # 更新源列表
ubuntu-drivers devices                               # 查看可安装的驱动列表（见下图选择recommended那项进行安装）
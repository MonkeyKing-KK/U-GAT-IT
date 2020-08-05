# U-GAT-IT

## INTRODUCTION
Image-to-image translation旨在学习一个在两个不同域里映射图像的函数, 其应用包括image inpainting(图像修复), super resolution(超分辨率), colorization和style transfer(风格迁移)<br>
当成组样本给定时,可以通过有监督的conditional generative model或者简单的regression model来训练映射模型.而在无监督情况下,已有很多工作完成了使用shared latent space和cycle consistency assumptions来translate images. <br>
这些方法仍有性能上的差异,具体取决于域间形状和纹理的变化量上.比如,它们可以很好完成纹理的映射但对于形状变化较大的时候就显得力不从心.因此需要一些预处理操作如图像裁剪和对齐来限制数据分布的复杂性.<br>
我们提出了一种无监督的图像到图像翻译的新方法,该方法以端到端的方式结合了新的注意力模块和新的可学习的归一化函数.模型基于辅助分类器获得注意力图,通过区分源域和目标域,指导翻译专注于更重要的区域,而忽略次要区域.<br>
这些注意力图在生成器和判别器中都有嵌入来专注于重要的语义区域,促进形状变换.生成器中的注意力图引起了对专门区分这两个域的区域的关注,而鉴别器中的注意力图则通过关注目标域中真实图像和伪图像之间的差异来帮助进行微调.<br>
提出了自适应层实例归一化化（AdaLIN），其参数是在训练期间通过在实例归一化（IN）和层归一化（LN）之间选择合适的比率从数据集中学习的.<br>
因此,无需修改模型架构或超参数，就可以完成较大转变的image translation任务.

## CFG-GAN in pyTorch

This is the author implementation of CFG-GAN (composite functional gradient learning of GAN), described in "*A framework of composite functional gradient methods for generative adversarial models*" [[Johnson & Zhang, TPAMI]](http://riejohnson.com/paper/cfggan-tpami.pdf), which is a long version of "*Composite functional gradient learning of generative adversarial models*" [[Johnson & Zhang, ICML 2018]](https://arxiv.org/abs/1801.06309).  

**_Requirements_**  
* Python version 3
* Tested with pyTorch 1.2.0 and torchvision 0.4.0. 
* pip install -r requirements.txt

**_Examples of image generator training_** 

By default, training examples here all periodically generate a collage of a few images.  

* To train on MNIST with its default discriminator/approximator DCGAN extension

        python3 train_cfggan.py --dataset MNIST
        
* To train on 64x64 LSUN bedroom with its default discriminator/approximator 4-block ResNet

  First, download the LSUN bedroom data as instructed at [https://github.com/fyu/lsun](https://github.com/fyu/lsun) and extract `bedroom_train_lmdb.zip` at `lsun-root`, and then 

        python3 train_cfggan.py --dataset lsun_bedroom64 --dataroot lsun-root
               
  (Or try out a smaller LSUN class `church_outdoor` in place of `bedroom` above.)               
               
* To save models after 100 stages, 200 stages, ... with filenames starting with `mod/MNIST-dcganx`

        mkdir mod
        python train_cfggan.py --dataset MNIST --save_interval 100 --save mod/MNIST-dcganx
                       
* To get help

        python3 train_cfggan.py -h

* **_Note_** `--num_stages` specifies how long training should go on.  The default value is set very large, perhaps much larger than necessary, and so please stop training once the quality of generated images hits a plateau by keyboard interruption, or set `--num_stages` smaller.  If training goes on very long, the model could eventually get to the  state of overfitting or overtraining.  

**_Examples of image generation_**

* To generate images from a saved model `mod/MNIST-dcganx-stage01000.pth`

        python3 cfggan_gen.py --saved mod/MNIST-dcganx-stage01000.pth --gen gen/MNIST-dcganx
        
  This generates a collage of 40 images (default) and write it with a filename starting with `gen/MNIST-dcganx`.        

* To get help

        python3 cfggan_gen.py -h        

**_Notes_** 

* The code uses a GPU whenever it is available.  To avoid use of GPUs even when it is available, 
  empty `CUDA_VISIBLE_DEVICES` via shell before calling python.  
  
        export CUDA_VISIBLE_DEVICES=""      

**_References_**

[[Johnson & Zhang, TPAMI]](http://riejohnson.com/paper/cfggan-tpami.pdf) A framework of composite functional gradient methods for generative adversarial models. Rie Johnson and Tong Zhang.  Accepted for publication in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) in 2019.     
[[Johnson & Zhang, ICML 2018]](https://arxiv.org/abs/1801.06309) Composite functional gradient learning of generative adversarial models.  Rie Johnson and Tong Zhang.  ICML 2018.  

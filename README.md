# chroma-pool
new pooling mehod designed for CNN for audio spectrogram  
see [chroma.py](https://github.com/exeex/chroma-pool/blob/master/chroma.py)    

This is a STFT spectrogram.
![](https://github.com/exeex/chroma-pool/raw/master/results/32367754_1621887231194499_6044579815242072064_n.png)

As shown above, horizontal lines split pixels to different areas along the frequency axis, which
 are corresponding to a frequency intervals of the piano keys:  
![](https://www.yamaha-keyboard-guide.com/images/xpiano_keys_and_notes.png.pagespeed.ic.N3bL4U9rPn.png)


So far I do is apply chroma pool on the RAW STFT, The result is shown below:    
![](https://raw.githubusercontent.com/exeex/chroma-pool/master/results/32653013_1621925511190671_1258526267855077376_n.png)


# Next to do:  
Design a CNN Model  
1. to remove harmonics (split to different channels)  
2. output pure piano roll 
3. split details to different channel

# Dialect_classification
MFCC based British English dialects classification

# ASR
There is a major misconception about dialects and accents being interchangeable words. Dialects are variations in the wordings, the grammar and the pronunciations in the same language whereas accents are just variations in pronunciations. Different dialects and accents arise due to a lot of factors like age, gender, culture, and locality. 

The variations in wordings for different dialects has caused many ASR applications to often fail in recognizing user’s speech. This has encouraged people to find ways to distinguish dialects. Different grammar and word selection in dialects force us to understand how each dialect use phonemes.  Phonemes are what differentiate words from one other in a particular dialect/language. By finding patterns in phonemes, we can possibly try classifying them. 

## Approach
Because phonemes are part of words, we need to break down our audio samples into small parts for analysis. For this, we break down the audio signal into small parts which approximately contain only one phoneme. Each small part would be one frame. Statistically, the length of one frame is usually around 25ms. Anything more or less than this makes the spectral content of the frame ambiguous. While breaking down our audio sample into frames, we make sure there is some overlap present between them to find a correlation between adjacent phonemes. This overlap is called stride (just like the stride in convolutional neural networks). Again, statistically, this stride is taken to be around 15ms. Anything less won’t be useful and anything more would increase overfitting. This is done because, in dialects, the temporal positioning of phonemes in speech also plays a huge role. A phoneme may change to some other phoneme when placed adjacent to some other phoneme. 

We’ve broken down our audio samples into frames but the issue here is that the audio signal is broken abruptly in the time domain, which will give rise to an infinite spectrum when we do our spectral analysis. To avoid this, we apply a Hamming window to the frames to smoothen out the frame endings. 

Now that our frames are ready for spectral analysis, we find the power spectrum of each frame to analyze frequency content in each phoneme. 

But instead of analyzing the bare spectrum, we can scale the spectrum to match our perceptual frequency ranges. Humans can hear frequencies between 20 to 20KHz, but our hearing is more discriminative at lower frequencies than at higher frequencies. We can distinguish the minute changes in frequency at lower frequencies but not higher frequencies. So obviously analyzing higher frequencies with a much wider range makes more sense to catch phoneme based features. To convert our power spectrum range from the generic range to a more perceptual range, we use something called as the Mel scale. Mel scale normalizes the frequency scale to match our perceptual frequency distinguishing capabilities. This can be seen in the following table. 

We see that an increment of around 240Hz, from 160Hz to 394Hz is equivalent to 250 Mels and the same jump of 250 Mels at higher frequencies is a jump of 5000Hz from 9000Hz to 14000Hz. 

But how do we use the Mel scale to create a new spectral picture of the frame? We use filter banks for this. 

Filter banks are nothing but triangular filters which when multiplied with the original frame spectrum, gives us a new spectral picture. We use the Mel scale to appropriately select the widths of the filter banks. The filter bank triangle starts from Mel scale value m, peaks at m+1 and then comes down to zero at m+2. Next filter bank starts at m+1 instead of m+2, this is to avoid blindspots in our new spectral analysis. You can see that in this way, at higher frequencies, the filter bank width is big, this is because our hearing and speech generation works poorly at higher frequencies, hence our filter will take a bigger range to accommodate changes in the spectrum. 

We then multiply our filter bank with the spectrum of each frame to find patterns in different frequency ranges. This is then done with all the frames generated from the audio sample. We convert filtered frame outputs into decibels and apply a de-correlation transform like DCT (discrete cosine transform) to remove correlations between frames. This is done because machine learning algorithms sometimes fail when there is a heavy correlation. The DCT coefficients are almost statistically independent. DCT removes higher frequency components and this is important because, in most speech data, information mostly resides in the lower frequency parts than the higher frequency parts. It is also the shape of the spectrum which is more important than the actual values. The coefficients after applying DCT are called log MFSC (Mel frequency spectral coefficients). We take the first 12 log MFSC coefficients because the high-frequency data doesn’t help us much in dialect classification. This is because most of the human speech spectrum is at lower frequencies. 

Till now we’ve worked on individual phoneme spectral content but dialects also have different velocities and acceleration of transition between phonemes. You might have noticed how the speech is faster in IDR3 compared to speech in IDR2. We can create delta coefficients (velocity) and delta-delta coefficients (acceleration) to learn these features. 

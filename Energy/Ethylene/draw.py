import matplotlib.pyplot as plt
import numpy as np

def draw_picture(type, accuracy, loss):
    length = len(accuracy)
    x = np.arange(1, length+1, 1)
    plt.subplot(2, 1, 1)
    plt.plot(x, accuracy, "yo-")
    plt.title('accuracy of {}'.format(type))
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.grid()


    plt.subplot(2, 1, 2)
    plt.plot(x, loss, "r.-")
    plt.title('loss of {}'.format(type))
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid()
    plt.savefig("../pictures/{}.png".format(type))
    plt.show()
#Fig.8
def draw_test():
    elm = [0.0634,0.0392,0.0429,0.0432,0.0445,0.0437,0.0551,0.0824]
    bp = [0.0753,0.0748,0.0615,0.0676,0.0635,0.0641,0.0691,0.0717]
    x = np.arange(5, 5+len(elm)*5, 5)
    plt.plot(x, elm, 'yo-', label='CF-ELM network')
    plt.plot(x, bp, 'r.-', label='CF-BP network')
    plt.legend()
    plt.xlabel('Number of hidden layer nodes')
    plt.ylabel('Generalization error')
    plt.show()

#Fig.9
def normalize(input):
    result = []
    for i in range(len(input)):
        result.append(input[i][0])
    return result
#11,12,13 -1
#-5 -0.5
def draw_models_result():
    bp = [[158143.98],
          [173393.89],
          [154034.66],
          [167378.4 ],
         [171920.83],
         [173290.53],
         [176029.  ],
         [171880.98],
         [158177.7 ],
         [158449.45],
         [62430.42],
         [69320.38],
         [54034.66],
         [157814.9 ],
         [170196.64],
         [159574.3 ],
         [158278.8 ],
         [166279.1 ],
         [168024.34],
         [170211.34],
         [128041.39],
         [168151.39],
         [154034.66],
         [161035.  ],
         [164684.45]]
    elm = [[166478.8 ],
         [168670.73],
         [168292.8 ],
         [164925.45],
         [169199.7 ],
         [167101.72],
         [163488.14],
         [168448.84],
         [166281.23],
         [165558.17],
         [64790.48],
         [69952.17],
         [67627.55],
         [158908.88],
         [160394.8 ],
         [167466.83],
         [163465.73],
         [158735.53],
         [168176.84],
         [164008.28],
         [119987.  ],
         [163488.14],
         [163044.27],
         [159371.69],
         [168770.53]]
    cnn = [[163138.61],
         [163958.55],
         [166631.56],
         [166555.48],
         [166583.83],
         [167917.36],
         [170626.6 ],
         [170822.56],
         [165349.34],
         [161930.42],
         [65578.1 ],
         [62937.4 ],
         [60503.45],
         [163258.4 ],
         [165349.34],
         [169127.62],
         [166644.5 ],
         [158349.92],
         [162485.84],
         [165349.34],
         [115349.34],
         [169838.31],
         [170120.52],
         [165349.34],
         [165349.34]]
    label = [[161539.4],
         [160770.5],
         [170118. ],
         [162793.4],
         [174664.1],
         [173315. ],
         [177995.1],
         [175180.9],
         [168093. ],
         [166978.3],
         [61731. ],
         [64328.1],
         [66604.3],
         [161503. ],
         [163113.1],
         [170770.1],
         [168170. ],
         [156113. ],
         [164451. ],
         [167863.9],
         [118750. ],
         [172278. ],
         [174615.8],
         [168923. ],
         [169317.4]]
    bp = normalize(bp)
    elm = normalize(elm)
    cnn = normalize(cnn)
    label = normalize(label)
    x = np.arange(1,len(bp)+1,1)
    plt.plot(x, bp, 'g.-', label='CF-BP network')
    plt.plot(x, elm, 'yo-', label='CF-ELM network')
    plt.plot(x, cnn, 'b.-', label='CF-1DCNN network')
    plt.plot(x, label, 'r.-', label='real_data')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.show()
#draw_models_result()
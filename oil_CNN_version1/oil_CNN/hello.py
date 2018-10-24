from flask import Flask
from flask import render_template
from flask import flash,request,session
from oil_CNN import oil_cnn
from data_utils import util
from draw import draw
import tensorflow as tf

app = Flask(__name__)
U = util()
X_, y_ = U.read_data('test')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify')
def classify():
    result = []
    for i in range(0,len(X_)):
        t = {}
        t['id'] = i
        t['data'] = X_[i]
        if y_[i] == 1:
            t['type'] = '中水淹层'
        elif y_[i] == 2:
            t['type'] = '弱水淹层'
        else:
            t['type'] = '强水淹层'
        result.append(t)

    return render_template('classify.html',result=result)

@app.route('/classifytwo/<rid>')
def classifytwo(rid = 0):
    #绘制波形图并保存
    d = draw()
    d.data = X_[int(rid)]
    d.draw_picture(rid)
    #返回波形图的地址并传递
    return render_template('classifytwo.html',rid=rid)

@app.route('/classifythree/<rid>')
def classifythree(rid = 0):
    return render_template('classifythree.html',rid=rid)

@app.route('/classifyfour/<rid>')
def classifyfour(rid = 0):
    oil = oil_cnn()
    graph = tf.Graph()
    prob = oil.exec(graph,X_[int(rid)],y_[int(rid)])
    print(prob)
    label = oil.judge(y_[int(rid)])
    prob = list(prob)
    pre = prob.index(max(prob)) + 1
    predict = oil.judge(pre)
    if predict == label:
        result = '正确'
    else:
        result = '错误'
    return render_template('classifyfour.html',label=label,predict=predict,result=result,prob=prob)

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/traintwo', methods=['GET', 'POST'])
def traintwo():
    form = request.form
    form_batch_size = int(form['batch_size'])
    form_epochs = int(form['epochs'])
    form_learning_rate = float(form['learning_rate'])
    form_keep_prob = float(form['keep_prob'])

    oil = oil_cnn()
    graph = tf.Graph()
    oil.batch_size = form_batch_size
    oil.epochs = form_epochs
    oil.keep_prob = form_keep_prob
    oil.learning_rate = form_learning_rate

    s = str(form_epochs)+str(form_batch_size)
    oil.ex(graph,s)
    return render_template('traintwo.html',s=s)

@app.route('/return_firstpage')
def return_firstpage():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

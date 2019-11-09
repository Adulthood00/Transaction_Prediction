import pandas as pd
import tensorflow as tf

path2="~/Downloads/santander-customer-transaction-prediction/test.csv"

test=pd.read_csv(path2)
test.head()
ID_code=test["ID_code"]
ID_code.head()
test=test.drop(["ID_code"],axis=1)
test.head()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(test)
test=scaler.transform(test)



with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_deep_santa_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))

graph=tf.get_default_graph()
graph.get_operations()[100:200]



X=graph.get_tensor_by_name("X:0")

#accuracy=graph.get_tensor_by_name("Cast:0")
#correct=graph.get_tensor_by_name("in_top_k/InTopKV2/k:0")
softmax=graph.get_tensor_by_name("santa_dnn/y_proba:0")
#logits=graph.get_tensor_by_name("dense_3/kernel/Initializer/random_uniform/shape:0")
feed_dict = {X:test}


with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_deep_santa_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
  #predictions=sess.run(correct,feed_dict)
  predictions=sess.run(softmax,feed_dict)
  #predictions=correct.eval(feed_dict)

  
finals=[]
for i in range(len(predictions)):
    if predictions[i][0]>predictions[i][1]:
        finals.append(0)
    else:
        finals.append(1)
        
    


ID=[]
for i in ID_code:
    ID.append(i)


with open("submission.txt","w") as file:
    for i in range(len(ID)):
        wr="{},{}\n".format(ID[i],finals[i])
        file.write(wr)




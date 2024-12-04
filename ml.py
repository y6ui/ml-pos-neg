import matplotlib.pyplot as plt
import numpy as np
import math
import random
def activation_func(nodes):
    return np.maximum(0,nodes)

def output_activation_func(nodes):
    nodes=nodes.copy()
    sum_nodes=np.sum(nodes)
    max_node=np.max(nodes)
    nodes-=max_node
    for i in range(len(nodes)):
        nodes[i]=math.e**nodes[i]
    sum_nodes=np.sum(nodes)
    for i in range(len(nodes)):
        nodes[i]/=sum_nodes
    return nodes

def calc_loss(end_nodes,ideal_output):
    sum=0
    end_nodes=end_nodes.nodes
    for i in range(len(end_nodes)):

        #sum+=(np.log(0.0000001+abs(end_nodes[i]-ideal_output[i])))
        sum+=(np.log(abs((end_nodes[i]))))*ideal_output[i]
        #sum+= math.sqrt((abs(end_nodes[i]-ideal_output[i])))
    return sum*(-1)/len(ideal_output)

class nodes:
    def __init__(self,num_inputs,num_nodes):
        self.weights=np.random.randn(num_inputs,num_nodes)
        self.bieses=np.zeros((num_nodes))
    def calculate(self,inputs):
        self.nodes=np.dot(inputs,self.weights)+self.bieses
def run(inputs,hidden_layers,output_layer):
    hidden_layers[0].calculate(inputs)

    for i in range(len(hidden_layers)-1):
        hidden_layers[i+1].calculate(hidden_layers[i].nodes)
    output_layer.calculate(hidden_layers[-1].nodes)
    output_layer.nodes=output_activation_func(output_layer.nodes)
def create_hidden_layers(num_inputs,amount_of_layers,amount_of_nodes_in_layer):
    layers=[]
    layer=nodes(num_inputs,amount_of_nodes_in_layer)
    layers.append(layer)
    for i in range (amount_of_layers-1):
        layer=nodes(amount_of_nodes_in_layer,amount_of_nodes_in_layer)
        layers.append(layer)
    return layers

def change_weight(layer_weights,i,j):
    layer_weights_t=layer_weights.copy()
    layer_weights_t[i][j]+=random.uniform(-0.04,0.04)
    return layer_weights_t
def change_bieses(layer_bieses,i):
    layer_bieses_t=layer_bieses.copy()
    layer_bieses_t[i]+=random.uniform(-0.004,0.004)
    return layer_bieses_t
def get_ideal_output(inputs):#waiting.....

    if(inputs[0]>0 ):
        ideal_output=[1,0]
    else:
        ideal_output=[0,1]

    return ideal_output
def change_inputs(inputs):

    return get_inputs()
def get_inputs():#waiting.....
    return [random.uniform(-10,10)]
def main():
    num_of_inputs=1
    num_of_output=2
    num_of_hidden=3
    num_of_nodes_in_hidden=4
    hidden_layers=create_hidden_layers(num_of_inputs,num_of_hidden,num_of_nodes_in_hidden)
    output_layer=nodes(num_of_nodes_in_hidden,num_of_output)
    x = [0]
    y = [0]
    inputs=get_inputs()
    #inputs=get_inputs()
    ideal_output=get_ideal_output(inputs)

    run(inputs,hidden_layers,output_layer)
    if(calc_loss(output_layer,ideal_output)>=0):
        min_loss=calc_loss(output_layer,ideal_output)
    else:
        min_loss=100
    for i in range(10000):
        

        ideal_output=get_ideal_output(inputs)

        temp_loss=min_loss
        for layer_num in range(num_of_hidden):
            temp_hidden_layer_weights=hidden_layers[layer_num].weights.copy()
            temp_hidden_layer_bieses=hidden_layers[layer_num].bieses.copy()
            

            for ii in range(len(hidden_layers[layer_num].weights)):

                for j in range(len(hidden_layers[layer_num].weights[0])):
                    inputs=get_inputs()
                    t_temp_hidden_layers_weights=change_weight(temp_hidden_layer_weights,ii,j)
                    t_temp_hidden_layers_bieses=change_bieses(temp_hidden_layer_bieses,j)
                    
                    hidden_layers[layer_num].weights=t_temp_hidden_layers_weights
                    hidden_layers[layer_num].bieses=t_temp_hidden_layers_bieses
                    ideal_output=get_ideal_output(inputs)
                    run(inputs,hidden_layers,output_layer)
                    loss1=calc_loss(output_layer,ideal_output)
                    inputs=change_inputs(inputs)
                    ideal_output=get_ideal_output(inputs)
                    run(inputs,hidden_layers,output_layer)
                    
                    loss2=calc_loss(output_layer,ideal_output)
                    if((loss1-0.01<=min_loss and loss2-0.01<=min_loss )and (loss1>=0 and loss2>=0 )):
                        if(min_loss==0):
                            pass
                        else:
                            x.append(x[-1]+1)
                            if(1-min_loss<0):
                                y.append(0) 
                            else:
                                y.append(1-min_loss)
                        min_loss=loss2
                        
                    elif((((loss1>=min_loss or loss2>=min_loss) )or (loss1<0 or loss2<0) )):
                        hidden_layers[layer_num].weights=temp_hidden_layer_weights
                        hidden_layers[layer_num].bieses=temp_hidden_layer_bieses

                        
                        
        
        temp_output_layer_weights=output_layer.weights.copy()
        temp_output_layer_bieses=output_layer.bieses.copy()
        for ii in range(len(output_layer.weights)):
            for j in range(len(output_layer.weights[0])):
                inputs=get_inputs()
                t_temp_output_layer_weights=change_weight(temp_output_layer_weights,ii,j)
                t_temp_output_layer_bieses=change_bieses(temp_output_layer_bieses,j)
                output_layer.weights=t_temp_output_layer_weights
                output_layer.bieses=t_temp_output_layer_bieses
                ideal_output=get_ideal_output(inputs)
                run(inputs,hidden_layers,output_layer)
                loss1=calc_loss(output_layer,ideal_output)
                inputs=change_inputs(inputs)
                run(inputs,hidden_layers,output_layer)
                ideal_output=get_ideal_output(inputs)
                loss2=calc_loss(output_layer,ideal_output)
                if((((loss1+0.01>=min_loss or loss2+0.01>=min_loss) )or(loss1<0 or loss2<0) )or not(loss1>=0 and loss2>=0 )):
                    output_layer.weights=temp_output_layer_weights
                    output_layer.bieses=temp_output_layer_bieses
                else:
                    if(min_loss==0):
                        pass
                    else:

                        x.append(x[-1]+1)
                        if(1-min_loss<0):
                            y.append(0) 
                        else:
                            y.append(1-min_loss)
                    #print("from: "+str(min_loss)+", to: "+str(loss2))
                    min_loss=loss2
        
        print("percentege: "+str((1-min_loss)*100))


        
    plt.plot(x, y)
    plt.show()
    inputs=[0]

    while(True):

        num1=float(input("give a num1 "))
        run([num1],hidden_layers,output_layer)
        print("resualt: ["+str(output_layer.nodes[0])+","+str(output_layer.nodes[1])+"]")
        if(max(output_layer.nodes)==output_layer.nodes[0]):
            print("you wrote a positive num")
        else:
            print("you wrote a negative or 0 num")
        





main()

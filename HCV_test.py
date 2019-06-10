
# coding: utf-8

# In[11]:


import networkx as nx
import random as rnd
import numpy as np
from random import choice
from scipy.stats import nbinom
from networkx.generators.community import stochastic_block_model
import itertools as it
from scipy.stats import norm
from scipy.stats import beta
from numpy import prod
import math
import os
import pickle


# these two functions save and load resp. .pkl files:

def save_obj(obj, name):
    with open('Python/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('Python/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def energy(graph):
    to_sum = []
    for edge in graph.edges():   
        a = -1 + 2*graph.nodes[edge[0]]['health']  # maps 0,1 to -1,1
        b = -1 + 2*graph.nodes[edge[1]]['health']
        to_sum.append(-a*b)
    return sum(to_sum)

def sick(p_sick):           # this will return 1 if dice roll is less than the input p_sick, returns 0 otherwise
    if rnd.random()<p_sick:
        return 1
    else:
        return 0

def BETA(mean,var):   # returns the a and b params for the beta dist. given mean and variance
    a = mean*(mean*(1-mean)-var)/var
    b = (1-mean)*(mean*(1-mean)-var)/var
    return a,b

def candidate(graph,node):
    candidates = []
    for nde in graph.neighbors(node):
        candidates.extend([(nde,x) for x in graph.neighbors(nde) if graph.nodes[x]['block']==graph.nodes[nde]['block'] and x not in [node]+list(graph.neighbors(node))])
        
    return candidates
    
def clearance(alpha0,num,p0,p1,p_clear): # alpha0 gives the % of type 0 in num (1-alpha0 is the % of type 1), p0 gives the % of alpha0*num that has the clearane allele, p1 gives the corresponding % of (1-alpha0)*num; all are randomly put into an array of len = num
    if num>1:
        n_0 = int(alpha0*num)        # number of type 0
        n_1 = num - int(alpha0*num)  # number of type 1

        n_00 = n_0 - int(p0*n_0)     # num of type 0 w/o clearance allele
        n_01 = int(p0*n_0)           # num of type 0 w/ clearance allele
        n_10 = n_1 - int(p1*n_1)     # num of type 1 w/o clearance allele
        n_11 = int(p1*n_1)           # num of type 1 w/ clearance allele

        """ the first 0 or 1 indicates the group type """
        """ the second 0 or 1 indicates whether the clearance allele is present """
        """ if value in array is nonzero, clearance allele is present """

        a_00 = list(zip(np.zeros(n_00),[0]*n_00)) 
        a_01 = list(zip([p_clear]*n_01,[0]*n_01))  
        a_10 = list(zip(np.zeros(n_10),[1]*n_10)) 
        a_11 = list(zip([p_clear]*n_11,[1]*n_11))

        c = a_00+a_01+a_10+a_11
        rnd.shuffle(c)
        return c
    else:   # the case when num = 1
        if rnd.random()<alpha0:      # randomly decides if type 0
            if rnd.random()<p0:      # if agent has clearance capability
                return (p_clear,0) # type 0 w/ clearance capability
            else:
                return (0,0)       # type 0 w/o clearance capability
        elif rnd.random()<p1:    # if the agent is type 1 then if has clearance capability
            return (p_clear,1) # type 1 w/ clearance capability
        else:                    
            return (0,1)       # type 1 w/o clearance capability

def HCV_ABM_2(graph,positions,p_decay,p_birth,p_death,t_steps,no_blocks=2): # needles is a list of lists [a,b] where a = num of clean needles and b = num of infected needles
    
    birth = 0
    death = 0
    
    g = graph
    
#     n_meets = int(p_swap*len(g.edges()))   # num. of edges to randomly select at each time step
    
#     print('no. of meets = %s'%str(n_meets) + '\n' + 'no. of edges = %s'%str(len(g.edges())))
    
    Energy = []
    
#     N_sick = [len([x for x in g.nodes() if g.nodes[x]['health']==1])]
#     N_healthy = [len([x for x in g.nodes() if g.nodes[x]['health']==0])]
    N_sick_0 = [len([x for x in g if g.nodes[x]['health']==1 and g.nodes[x]['immunity_type']==0])]
    N_sick_1= [len([x for x in g if g.nodes[x]['health']==1 and g.nodes[x]['immunity_type']==1])]
    N_healthy_0 = [len([x for x in g if g.nodes[x]['health']==0 and g.nodes[x]['immunity_type']==0])]
    N_healthy_1 = [len([x for x in g if g.nodes[x]['health']==0 and g.nodes[x]['immunity_type']==1])]
    
    # keeps track of needles on an agent-level
#     needles = {}
#     for node in g.nodes():
#         needles.update({node:[(g.nodes[node]['health'],g.nodes[node]['needles'])]})

    #------------------------------------ this section for printing the network ------------------------------------#
    
#     graph_size = [15,9]
#     colors = []
    
#     for node in g.nodes():
#         if g.nodes[node]['health']==1:
#             colors.append('red')
#         else:
#             colors.append('blue')
        
#     needl_labls = {} # to display the needle quantities
#     for node in g.nodes():
#         needl_labls.update({node:g.nodes[node]['needles']})
    
#     node_names = {} # to display the node labels
#     for node in g.nodes():
#         node_names.update({node:node})
        
#     rand_pos = nx.spring_layout(g,pos=positions,fixed=positions.keys())
# #     rand_pos = positions  # uses the positions generated outside the function    
# #     rand_pos = nx.kamada_kawai_layout(g)  # aesthetic layout
# #     rand_pos = nx.spring_layout(g)   # another layout
# #     rand_pos = nx.spectral_layout(g,dim=2)   # yet another layout
    
#     labls_pos = {}  # needle label positions
#     for pt in rand_pos:
#         labls_pos.update({pt:rand_pos[pt]+np.array([0.05,0.05])}) # this will shift the labels a little to the right and up
        
#     nde_labl_pos = {}  # node label positions
#     for pt in rand_pos:
#         nde_labl_pos.update({pt:rand_pos[pt]-np.array([0.05,0.05])}) # this will shift the labels a little to the left and down
    
#     fig,ax = plt.subplots(figsize=graph_size)
#     plt.axis('off')
#     nx.draw_networkx(g,pos=rand_pos,node_color=colors,edgelist=[],with_labels=False,node_size=[300*g.nodes[nde]['p_use'] for nde in g.nodes()])
#     nx.draw_networkx_edges(g,pos=rand_pos,width=[5*g.edges[x,y]['wght'] for (x,y) in g.edges()])
#     nx.draw_networkx_labels(g,pos=labls_pos,labels=needl_labls)  # needle labels
#     nx.draw_networkx_labels(g,pos=nde_labl_pos,labels=node_names,font_weight='bold',font_size=15) # node labels
#     plt.show()
    
    #---------------------------------------------------------------------------------------------------------------#
    
    clean_ndls = [sum([g.nodes[nds]['needles'][0] for nds in g.nodes()])]  # initial num of clean needles
    infct_ndls = [sum([g.nodes[nds]['needles'][1] for nds in g.nodes()])]  # initial num of infected needles
    
    Energy.append(energy(g))
    
    degrees = []    # will house the number of edges in the graph over time
    
    edges_no = []

    for t in range(t_steps): # a time step represents the time scale associated with needle-exchanges of a given % of the population of users (controlled by the various probabilities)
        
        if rnd.random()<p_birth:   # if a new node is introduced
            
            """ we define some parameters for new nodes """
            a_break,b_break = BETA(0.01111,0.0001)  
            a_link,b_link = BETA(0.9,0.01)
            
            info = {'size':50,'immunity':(260,7000),'lend':(25,2),'accept':(25,2),'use':(5,5),'break':(a_break,b_break),'link':(a_link,b_link),'drug_a':(7,3),'drug_b':(5,5),'drug_c':(3,7)}
            
            node_num = 1+max(g)
            Immunity = clearance(0,1,p0,p1,0.00444) if rnd.random()<0.5 else clearance(1,1,p0,p1,0.00444) # randomly selects type 0 or 1
            immunity = Immunity[0]
            immunity_type = Immunity[1]
#             block_num = rnd.choice(range(no_blocks))
            block_num = immunity_type
#             info = group[names[block_num]]
            P_lend = beta(info['lend'][0],info['lend'][1]).rvs()
            P_accept = beta(info['accept'][0],info['accept'][1]).rvs()
            P_use = beta(info['use'][0],info['use'][1]).rvs()
            P_break = beta(info['break'][0],info['break'][1]).rvs()
            P_link = beta(info['link'][0],info['link'][1]).rvs()
            Drug_a = beta(info['drug_a'][0],info['drug_a'][1]).rvs()
            Drug_b = beta(info['drug_b'][0],info['drug_b'][1]).rvs()
            Drug_c = beta(info['drug_c'][0],info['drug_c'][1]).rvs()
            Age = int(norm(33,5).rvs())    # normal distribution of ages centered on 33 spread of 5
            Target = nbinom(3,0.6).rvs()   # target number of neighbors
            block_ndes = [nde for nde in g if g.nodes[nde]['block']==block_num]
            
            g.add_node(node_num,block=block_num,needles=[rnd.randint(1,3),0],health=0,immunity=immunity,immunity_type=immunity_type,p_lend=P_lend,p_accept=P_accept,p_use=P_use,p_break=P_break,p_link=P_link,drug_a=Drug_a,drug_b=Drug_b,drug_c=Drug_c,age=Age,target=Target)
            birth += 1

            
            if sum([g.degree(nde) for nde in block_ndes])>0:    # if the normalization is non-zero
                
                pop_node = np.random.choice(block_ndes,p=[g.degree(nde)/sum([g.degree(nde) for nde in block_ndes]) for nde in block_ndes]) # selected node in block based on degree popularity

#                 a1 = g.nodes[pop_node]['age']
#                 a2 = g.nodes[node_num]['age']
#                 b1 = g.nodes[pop_node]['immunity_type']
#                 b2 = g.nodes[node_num]['immunity_type']
    #             print(pop_node, node_num)

#                 if rnd.random()<1/(1+abs(a1-a2)+abs(b1-b2)):   # if a link is formed 
                a = g.nodes[pop_node]['p_use']
                b = g.nodes[node_num]['p_use']
                x = g.nodes[pop_node]['p_lend']
                y = g.nodes[node_num]['p_lend']
                dd = sum([g.nodes[node_num][drug]*g.nodes[pop_node][drug] for drug in ['drug_a','drug_b','drug_c']])

                g.add_edge(node_num,pop_node,color='k',wght=a*b*(x+y)*dd)      # create a new edge

#                     print('new edge at',(node_num,pop_node))

        nodes = list(g.nodes())
        rnd.shuffle(nodes)
        
        for node in nodes: 
            
            if g.nodes[node]['health']==1 and rnd.random()<g.nodes[node]['immunity']: 

                g.nodes[node]['health']=0 # the infected user recovers with a probability per time step

#                 print(node,'recovers')
                    
            if rnd.random()<g.nodes[node]['p_use']: # if user decides to inject
            
                if rnd.random()<(g.nodes[node]['p_lend']+g.nodes[node]['p_accept'])/2: # if user decides to share needles
                    
                    nhbrs = list(g.neighbors(node))
                    rnd.shuffle(nhbrs)
                    
                    for nhbr in nhbrs: # iterates thru neighbors of the node

                        if rnd.random()<g.edges[node,nhbr]['wght']/2: # we divide by 2 since each edge has 2 chances for an event to occur

                            g.add_edge(node,nhbr,color='r')


                            if g.nodes[node]['health']==g.nodes[nhbr]['health']: # like meets like case

                                if g.nodes[node]['health']==1: # this is the sick meets sick case

                                    x = nhbr if g.nodes[node]['p_lend']<g.nodes[nhbr]['p_lend'] else node # host with the higher p_lend will lend
      
                                    ax = g.nodes[x]['needles'][0]  # num of clean needles for user x
                                    bx = g.nodes[x]['needles'][1]  # num of infected needles for user x

                                    # modifying the % of clean needles in an infected user's collection
                                    if rnd.random()<ax/(ax+bx):  # chances that the needle is clean

                #                         print(edge[x],'infects a clean needle while sharing with',edge[1-x]) # turn on for description of events

                                        g.nodes[x]['needles'][0]-=1  # a clean needle is lost 
                                        g.nodes[x]['needles'][1]+=1  # an infected needle is gained
#                                         print('needle infected at',x)

                                else:  # this is the healthy meets healthy case

                                    x = nhbr if g.nodes[node]['p_lend']<g.nodes[nhbr]['p_lend'] else node # host with the higher p_lend, call it x, gives the needle, say
                       
                                    ax = g.nodes[x]['needles'][0]  # num of clean needles for user x
                                    bx = g.nodes[x]['needles'][1]  # num of infected needles for user x

                                    # healthy user injects first then shares (there's a chance the user has an infected needle)
                                    if rnd.random()<bx/(ax+bx):  # chances that the needle is infected

                #                         print(edge[x],'injects themselves with infected needle and lends to',edge[1-x]) # turn on for description of events

                                        g.nodes[nhbr]['health']=1    # both users
                                        g.nodes[node]['health']=1    # become infected
#                                         print(nhbr,'and',node,'become infected')


                            else: # healthy meets sick case

                                healthy_user = [user for user in [node,nhbr] if g.nodes[user]['health']==0][0]   # picks out the healthy one in the pair
                                sick_user = [user for user in [node,nhbr] if g.nodes[user]['health']==1][0]  # picks out the sick one in the pair

                                if  g.nodes[sick_user]['p_lend']<g.nodes[healthy_user]['p_lend']: # host with the higher p_lend gives the needle, say.

                                    a = g.nodes[healthy_user]['needles'][0]  # num of clean needles for user x
                                    b = g.nodes[healthy_user]['needles'][1]  # num of infected needles for user x

                                    if rnd.random()<a/(a+b): # if a clean needle was selected by the uninfected user

                                        g.nodes[healthy_user]['needles'][0]-=1  # uninfected user gives to infected (with needle return)
                                        g.nodes[healthy_user]['needles'][1]+=1  # and an infection of a clean needle occurs
#                                         print('needle infected at',healthy_user)
                                    else:

                #                         print(healthy_user,'injects with infected needle') # turn on for description of events

                                        g.nodes[healthy_user]['health']=1  # if selection of an infected needle occurs, the healthy user is assumed to use it before sharing, and therefore become infected
#                                         print(healthy_user,'becomes infected')
                                else:  # infected user gives needle to uninfected user (again, the lender is assumed to inject first then lend, in which case the healthy user becomes infected) 

                                    a = g.nodes[sick_user]['needles'][0]  # num of clean needles for user x
                                    b = g.nodes[sick_user]['needles'][1]  # num of infected needles for user x

                                    g.nodes[healthy_user]['health']=1 # healthy user becomes infected regardless of the type of needle selected

                                    if rnd.random()<a/(a+b): # if a clean needle is selected by the infected user:

                                        g.nodes[sick_user]['needles'][0]-=1  # uninfected needle becomes infected
                                        g.nodes[sick_user]['needles'][1]+=1  # from an injection by the infected user
#                                         print(healthy_user,'becomes infected and needle at',sick_user,'becomes infected')

    
            if rnd.random()<p_decay and g.nodes[node]['needles'][1]>0: 
                
#                 print('viral decay at node',node) # turn on for description of events
                
                g.nodes[node]['needles'][0]+=1   # with prob. p_decay (and if the user has any infected needles)
                g.nodes[node]['needles'][1]-=1   # an infected needle will "decay" to a clean one
#                 print('needle clears infection at',node)


            # the below section breaks/creates an edge based on whether node is below their target

            avlble_nodes = [nde for nde in g.nodes() if g.degree(nde)<g.nodes[nde]['target']] # list of available nodes

            if len(candidate(g,node))>0 and rnd.random()<g.nodes[node]['p_break']: # if node decides to break an edge
                s = sum([1/g.edges[node,nhbr]['wght'] for nhbr in [x[0] for x in candidate(g,node)]])
                wghts = [1/(s*g.edges[node,nhbr]['wght']) for nhbr in [x[0] for x in candidate(g,node)]] # this maps x -> 1/x (and normalizes) where x is one of n edge weights; this way probabilities are inverted
                break_index = np.random.choice(range(len(candidate(g,node))),p=wghts) 
                break_node,bond_node=candidate(g,node)[break_index]
                
                g.remove_edge(node,break_node)  # removes the edge
                
                a = g.nodes[node]['p_use']
                b = g.nodes[bond_node]['p_use']
                x = g.nodes[node]['p_lend']
                y = g.nodes[bond_node]['p_lend']
                dd = sum([g.nodes[node][drug]*g.nodes[bond_node][drug] for drug in ['drug_a','drug_b','drug_c']])

                g.add_edge(node,bond_node,color='k',wght=a*b*(x+y)*dd) 


#                 print('edge broken at',(node,break_node))

            if len(avlble_nodes)>1 and rnd.random()<g.nodes[node]['p_link'] and node in avlble_nodes:  # if there are at least two available nodes

                if len(list(g.neighbors(node)))>0:
        
                    next_nhbrs = []    # will house nodes one agent removed from node (i.e. two hops away)

                    for nde in g.neighbors(node):
                        next_nhbrs.extend(list(g.neighbors(nde)))

                    next_nhbrs = list(set(next_nhbrs))
                    next_nhbrs.remove(node)

                    for nde in next_nhbrs:
                        if nde in g.neighbors(node):
                            next_nhbrs.remove(nde)      # selects only those nodes which are two hops from node

                    avlble_nhbrs = [x for x in next_nhbrs if x in avlble_nodes]

                    if len(avlble_nhbrs)>0:

                        avlble_norm = sum([g.nodes[nde]['p_link'] for nde in avlble_nhbrs])

                        new_node = np.random.choice(avlble_nhbrs,p=[g.nodes[nde]['p_link']/avlble_norm for nde in avlble_nhbrs])   # choose a random available node

                        a = g.nodes[node]['p_use']
                        b = g.nodes[new_node]['p_use']
                        x = g.nodes[node]['p_lend']
                        y = g.nodes[new_node]['p_lend']
                        dd = sum([g.nodes[node][drug]*g.nodes[new_node][drug] for drug in ['drug_a','drug_b','drug_c']])

                        g.add_edge(node,new_node,color='k',wght=a*b*(x+y)*dd)      # create a new edge
#                         print('new edge at',(node,new_node))
                else:
                    
                    block_ndes = [nde for nde in g if g.nodes[nde]['block']==g.nodes[node]['block']]
                    
                    if sum([g.degree(nde) for nde in block_ndes])>0:    # if the normalization is non-zero
                        
                        pop_node = np.random.choice(block_ndes,p=[g.degree(nde)/sum([g.degree(nde) for nde in block_ndes]) for nde in block_ndes]) # selected node in block based on degree popularity

                        a1 = g.nodes[pop_node]['age']
                        a2 = g.nodes[node]['age']
                        b1 = g.nodes[pop_node]['immunity_type']
                        b2 = g.nodes[node]['immunity_type']
    #                     print(pop_node, node)

                        if rnd.random()<1/(1+abs(a1-a2)+abs(b1-b2)):   # if a link is formed 
                            a = g.nodes[pop_node]['p_use']
                            b = g.nodes[node]['p_use']
                            x = g.nodes[pop_node]['p_lend']
                            y = g.nodes[node]['p_lend']
                            dd = sum([g.nodes[node][drug]*g.nodes[pop_node][drug] for drug in ['drug_a','drug_b','drug_c']])

                            g.add_edge(node,pop_node,color='k',wght=a*b*(x+y)*dd)      # create a new edge
#                             print('new edge at',(node,pop_node))

            if g.nodes[node]['health']==1: # if user is infected
                    
                if rnd.random()<p_death and g.nodes[node]['immunity']==0:
                        
                    g.remove_node(node)     # agent leaves network
                    death += 1

#                     print(node,'leaves the network')
#                     del(positions[node])     # turn off when labelling is off
                        
                elif rnd.random()<g.nodes[node]['p_use']:
                        
                    a = g.nodes[node]['needles'][0] # num of clean needles in infected users collection
                    b = g.nodes[node]['needles'][1] # num of infected needles in infected users collection

                    if rnd.random()<a/(a+b): # chances that an uninfected needle is selected

#                         print('needle infected at node',node) # turn on for description of events

                        g.nodes[node]['needles'][0]-=1   # a new needle is infected 
                        g.nodes[node]['needles'][1]+=1   # with prob p_use*a/(a+b)                  
#                 print('no. available is',len(avlble_nodes))

        Energy.append(energy(g))
        
        clean_ndls.append(sum([g.nodes[nds]['needles'][0] for nds in g.nodes()])) # updates the num of clean needles
        infct_ndls.append(sum([g.nodes[nds]['needles'][1] for nds in g.nodes()])) # updates the num of infected needles
        
        # keeping track of populations at each time step:
        healthy_users = [x for x in g.nodes() if g.nodes[x]['health']==0]
        sick_users = [x for x in g.nodes() if g.nodes[x]['health']==1]
        
#         N_healthy.append(len(healthy_users))
#         N_sick.append(len(sick_users))

        N_sick_0.append(len([x for x in g if g.nodes[x]['health']==1 and g.nodes[x]['immunity_type']==0]))
        N_sick_1.append(len([x for x in g if g.nodes[x]['health']==1 and g.nodes[x]['immunity_type']==1]))
        N_healthy_0.append(len([x for x in g if g.nodes[x]['health']==0 and g.nodes[x]['immunity_type']==0]))
        N_healthy_1.append(len([x for x in g if g.nodes[x]['health']==0 and g.nodes[x]['immunity_type']==1]))

        
        # keeping track of the needles precisely:
#         for node in g.nodes():
#             needles[node].append((g.nodes[node]['health'],g.nodes[node]['needles']))
                                 
        #------------------------------------ this section for printing the network ------------------------------------#
#         colors = []
        
#         rand_pos = nx.spring_layout(g,pos=positions,fixed=positions.keys())
    
#         for node in g.nodes():
#             if g.nodes[node]['health']==1:
#                 colors.append('red')
#             else:
#                 colors.append('blue')
            
#         needl_labls = {}
#         for node in g.nodes():
#             needl_labls.update({node:g.nodes[node]['needles']})
            
#         labls_pos = {}  # needle label positions
#         for pt in rand_pos:
#             labls_pos.update({pt:rand_pos[pt]+np.array([0.05,0.05])}) # this will shift the labels a little to the right and up

#         nde_labl_pos = {}  # node label positions
#         for pt in rand_pos:
#             nde_labl_pos.update({pt:rand_pos[pt]-np.array([0.05,0.05])}) # this will shift the labels a little to the left and down

#         fig,ax = plt.subplots(figsize=graph_size)
#         plt.axis('off')
#         nx.draw_networkx(g,pos=rand_pos,node_color=colors,edgelist=[],node_size=[300*g.nodes[nde]['p_use'] for nde in g.nodes()],with_labels=False)
#         nx.draw_networkx_edges(g,pos=rand_pos,width=[5*g.edges[x,y]['wght'] for (x,y) in g.edges()],edge_color=[g.edges[x,y]['color'] for (x,y) in g.edges()])
#         nx.draw_networkx_labels(g,pos=labls_pos,labels=needl_labls) # needle labels
#         nx.draw_networkx_labels(g,pos=nde_labl_pos,labels=node_names,font_weight='bold',font_size=15) # node labels 
#         plt.show()
        #---------------------------------------------------------------------------------------------------------------#
        
        # this turns the red edges back to black at the end of each time step:
        for edge in g.edges():
            g.add_edge(edge[0],edge[1],color='k')
            
        degrees.append(sum([g.degree(nde) for nde in g])/2)
        
        
        female_edges = len([(x,y) for (x,y) in g.edges() if g.nodes[x]['block']+g.nodes[y]['block']==0])
        between_edges = len([(x,y) for (x,y) in g.edges() if g.nodes[x]['block']+g.nodes[y]['block']==1])
        male_edges = len([(x,y) for (x,y) in g.edges() if g.nodes[x]['block']+g.nodes[y]['block']==2])
    
        edges_no.append([female_edges,between_edges,male_edges])
            
#         print(len(list(g.edges())))

    N_healthy = [N_healthy_0,N_healthy_1]  # seperated by immunity type
    N_sick = [N_sick_0,N_sick_1]
        
    return (N_healthy,N_sick,clean_ndls,infct_ndls,Energy,degrees,edges_no,birth,death)



OR = 0.68  # np.linspace(0.58,0.81,30)
W,M = 0.5,0.5
C = 0.26   # np.linspace(0.2,0.35,30)
a = (OR - 1)*W
b = W + OR*M - (OR - 1)*C
c = -C
p0_a = (-b+np.sqrt(b**2 - 4*a*c))/(2*a)    # <-- turns out to be the value between 0 and 1
p0_b = (-b-np.sqrt(b**2 - 4*a*c))/(2*a)

p0 = p0_a                     # % of type 0 that have the clearance allele
p1 = OR*p0/(1+(OR-1)*p0)      # % of type 1 that have the clearance allele


a_break,b_break = BETA(0.01111,0.0001)
a_link,b_link = BETA(0.9,0.01)


names = ['A','B']
group = {}
for name in names:
    group.update({name:{'size':50,'immunity':(260,7000),'lend':(25,2),'accept':(25,2),'use':(5,5),'break':(a_break,b_break),'link':(a_link,b_link),'drug_a':(7,3),'drug_b':(5,5),'drug_c':(3,7)}})


N = len(names) # number of groups
p_sick = 0.05 # ratio of initially sick to total population
p_decay = 0.96
p_birth = 1/365
p_death = 0.005/365
t_steps = 3650


sizes = [group[x]['size'] for x in group] # sizes of blocks A, B, C (the more connections between the blocks, the less community structure)

ilessj = [(i,j) for (i,j) in it.product(range(N),range(N)) if i<j]

probs = np.zeros([N,N])

for i in range(N):
    probs[i,i] = beta(30,1000).rvs()
    
for i,j in ilessj:
    probs[i,j] = beta(20,1000).rvs()
    probs[j,i] = probs[i,j]
    
g = nx.stochastic_block_model(sizes, probs)

Needles = []
for i in range(sum(sizes)):
    Needles.append([rnd.randint(1,3),0])   # here we imagine that everyone starts off wih at least one clean needle


# saving the positions for the example graphs
# save_obj(nx.spring_layout(g),'temp_pos')    
positions = nx.spring_layout(g)

k = 0
Alpha = 0.5  # this quantifies the % of a community sampling from the high clearance distribution, 1-Alpha has the low
Beta = 0.5 # this quantifies the % of communites that have sample from the high clearance distribution, 1-Beta has the low


for name in names:
    size = group[name]['size']
    P_lend = beta(group[name]['lend'][0],group[name]['lend'][1]).rvs(size=size)
    P_accept = beta(group[name]['accept'][0],group[name]['accept'][1]).rvs(size=size)
    P_use = beta(group[name]['use'][0],group[name]['use'][1]).rvs(size=size)
    P_break = beta(group[name]['break'][0],group[name]['break'][1]).rvs(size=size)
    P_link = beta(group[name]['link'][0],group[name]['link'][1]).rvs(size=size)
    Drug_a = beta(group[name]['drug_a'][0],group[name]['drug_a'][1]).rvs(size=size)
    Drug_b = beta(group[name]['drug_b'][0],group[name]['drug_b'][1]).rvs(size=size)
    Drug_c = beta(group[name]['drug_c'][0],group[name]['drug_c'][1]).rvs(size=size)
    Age = [int(norm(33,5).rvs()) for i in range(size)]    # normal distribution of ages centered on 33 spread of 5
    Target = nbinom(5,0.6).rvs(size=size)   # target number of neighbors
    
    # turn on for heterogenous mixing determined by Alpha -------------- #
    Immunity = clearance(1,size,p0,p1,0.00444) if name=='A' else clearance(0,size,p0,p1,0.00444)   
    # ------------------------------------------------------------------ #
    
    # turn on for homogenous mixing determined by Beta ----------------- #
#     if rnd.random()<Beta:
#         Immunity = clearance(0,size,p0,p1,0.00444)  
#     else:
#         Immunity = clearance(1,size,p0,p1,0.00444)
    # ------------------------------------------------------------------ #
    
    i = 0
    for node in range(k,k+size):
        drug_norm = 1-P_use[i]+Drug_a[i]+Drug_b[i]+Drug_c[i]  # normalization for the probabilites
        drug_a = Drug_a[i]/drug_norm
        drug_b = Drug_b[i]/drug_norm
        drug_c = Drug_c[i]/drug_norm
        g.add_node(node,needles=Needles[node],health=sick(p_sick),immunity=Immunity[i][0],immunity_type=Immunity[i][1],p_lend=P_lend[i],p_accept=P_accept[i],p_use=P_use[i],p_break=P_break[i],p_link=P_link[i],drug_a=drug_a,drug_b=drug_b,drug_c=drug_c,age=Age[i],target=Target[i])
        i+=1
    k+=size
    
for edge in g.edges():
    a = g.nodes[edge[0]]['p_use']
    b = g.nodes[edge[1]]['p_use']
    x = (g.nodes[edge[0]]['p_lend']+g.nodes[edge[0]]['p_accept'])/2
    y = (g.nodes[edge[1]]['p_lend']+g.nodes[edge[1]]['p_accept'])/2
    dd = sum([g.nodes[edge[0]][drug]*g.nodes[edge[1]][drug] for drug in ['drug_a','drug_b','drug_c']])
    g.add_edge(edge[0],edge[1],color='k',wght=a*b*(x+y)*dd)   # here wght gives a measure of how often two uses share needles of the same drug type 

# save_obj(g,'temp_graph')

# fig,ax = plt.subplots(figsize=[15,9])
# nx.draw_networkx(g,pos=positions,edge_list=[],node_size=[300*g.nodes[nde]['p_use'] for nde in g.nodes()],with_labels=False,node_color=['C'+str(2*int(i)+2) for i in [g.nodes[nde]['immunity_type'] for nde in g.nodes()]])
# nx.draw_networkx_edges(g,pos=positions,width=[5*g.edges[x,y]['wght'] for (x,y) in g.edges()],edge_color=[g.edges[x,y]['color'] for (x,y) in g.edges()])
# ax.axis('off')
# plt.savefig('network types')

save_obj(g,'initial graph')

(N_healthy,N_sick,clean_ndls,infct_ndls,Energy,Degrees,Edges_no,birth,death) = HCV_ABM_2(g,positions,p_decay,p_birth,p_death,t_steps)

pops = [N_healthy,N_sick,clean_ndls,infct_ndls,Energy,Degrees,Edges_no,birth,death]
save_obj(pops,'pops 1')
save_obj(g,'final graph')


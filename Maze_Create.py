import numpy as np
import cv2
import random

class maze_creator:
    # Initialization function
    def __init__(self):
        self.mazes = []
        self.current_maze = []
        self.mazes_amount = 0
        self.current_index = 0
        self.current_width = 0
        self.current_height = 0
        self.dimensions = []
        self.valid_symbols = ["#","S"," ", "", "2", "4", "8"]

    # Create maze of specific size function
    def create_maze_size(self,width,height):
        maze = []
        self.dimensions.append([width,height])
        for col in range(height):
            row = [" "]*width
            maze.append(row)
        self.mazes.append(maze)
        self.mazes_amount = self.mazes_amount

    # Select current maze
    def select_maze(self,index):
        self.current_maze = self.mazes[index]
        self.current_index = index
        self.current_width = self.dimensions[index][0]
        self.current_height = self.dimensions[index][1]

    # Visualize maze
    def visualize_maze(self, border):
        for row in self.current_maze:
            if border == 1:
                print("-"*(2*len(row)+1))
                print("|",end="")
            for value in row:
                if border == 1:
                    print(value, end="|")
                else:
                    print(value,end="")
            print("")
     
    def visualize(self,maze):
        for row in maze:
            for value in row:
                print(value,end="")
            print("")
    
    # Insert cheese function (coordinates in x,y)
    def insert_cheese(self,amount,coordinates):
        cheese_list = []
        index = 0
        for cheese in amount:
            x = coordinates[index][0]
            y = coordinates[index][1]
            try:
                self.current_maze[y][x] = cheese
                cheese_element = [cheese,x,y]
                cheese_list.append(cheese_element)
            except:
                print("coordinates at index %i does not match maze size" %index)
            index = index+1

    # suround maze with walls
    def insert_edge_walls(self):
        index = 0
        while index < self.current_height:
            if index == 0 or index == self.current_height-1:
                col = 0
                while col < self.current_width:
                    self.current_maze[index][col] = "#"
                    col = col+1
            else:
                self.current_maze[index][0] = "#"
                self.current_maze[index][self.current_width-1] = "#"
            index = index + 1

    # Create walls section
    def insert_walls(self,coordinates):
        if len(coordinates) > 1:
            for location in coordinates:
                x = location[0]
                y = location[1]
                try:
                    self.current_maze[y][x] = "#"
                except:
                    print("coordinates %i ,%i does not match maze size" %(x,y))
    
    # Set spawn point for current maze
    def insert_spawn(self,coordinate):
        x = coordinate[0]
        y = coordinate[1]
        self.current_maze[y][x] = "S"
    
    # get maze from matrix
    def insert_maze(self,maze):
        # Check if maze is valid
        for row in maze:
            for element in row:
                element_valid = 0
                for valid in self.valid_symbols:
                    if element == valid:
                        element_valid = 1
                        break
                if element_valid == 0:
                    print("invalid symbol %s" %element)
                    return
        self.dimensions.append([len(maze[0]),len(maze)])
        self.mazes.append(maze)
    
    # return desired maze
    def get_maze(self,index):
        try:
            return self.mazes[index]
        except:
            print("Index out of range")

class maze_sarsa:
    def __init__(self):
        # map parameters
        self.map = []
        self.original_map = []
        self.width = 0
        self.height = 0

        # state parameters
        self.current_state = [0,0]
        self.all_actions = []
        self.initial_state = [0,0]

        # algorithm parameters
        self.epsilon = 0 
        self.learning_rate = 0
        self.Q = 0
        self.gamma = 0
        self.move_penalty = 0
        self.total_score = 0
    
    def give_score(self,score):
        self.total_score = self.total_score+score
    def get_score(self):
        return self.total_score
    def reset_score(self):
        self.total_score = 0
    
    def remove_reward(self,state):
        self.map[state[1]][state[0]] = " "

    def set_map(self,maze):
        self.original_map = maze
        self.map = np.copy(self.original_map)
        self.width = len(maze[0])
        self.height = len(maze)
    
    def load_Q_from_txt(self,filename,actions_num = 4):
        self.Q = np.zeros((self.width*self.height,actions_num))
        file = open(filename,"r")
        lines = file.readlines()
        index = 0
        for line in lines:
            action_index = 0
            for value in line.split(","):
                self.Q[index,action_index] = float(value)
                action_index = action_index+1
            index = index+1

    def reset_map(self):
        self.map = np.copy(self.original_map)
        self.current_state = self.initial_state

    def initialize_parameters(self,epsilon=0.1, move_penalty = -0.1, learning_rate=0.2,gamma=0.9, actions_num = 4,state = [0,0],actions =["up","down","left","right"]):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.current_state = state
        self.initial_state = state
        self.Q = np.zeros((self.width*self.height,actions_num))
        # randomize Qtable with values between 0 and 1
        for position in range(len(self.Q)):
            for action in range(len(self.Q[position])):
                self.Q[position][action] = random.uniform(0,1)
        self.all_actions = actions
        self.move_penalty = move_penalty

    def set_initial_state(self,state):
        self.initial_state = state
        self.current_state = state

    def check_identical_weights(self,x,y):
        index = y*self.width+x
        for weight in self.Q[index,:]:
            if weight != self.Q[index,0]:
                return False
        return True
    
    def select_action(self,state):
        x = state[0]
        y = state[1]

        # determine possible actions
        possible_actions = []
        if self.map[y-1][x] != "#":
            possible_actions.append("up")
        if self.map[y+1][x] != "#":
            possible_actions.append("down")
        if self.map[y][x-1] != "#":
            possible_actions.append("left")
        if self.map[y][x+1] != "#":
            possible_actions.append("right")

        # Check if all action weights are identical
        equal = self.check_identical_weights(x,y)
        index = y*self.width+x

        # Check if random action should be taken
        if np.random.uniform(0,1) < self.epsilon or equal == True:
            action = np.random.choice(possible_actions)
        # else take highest ranked action at current state
        else:
            action = ""
            while possible_actions.count(action) == 0:
                # need to take the correct action vector from the width*height possible
                index = y*self.width+x
                # get values of possible actions
                possible_indexes = []
                action_values = []
                for poss_index in range(len(possible_actions)):
                    pos_index = self.all_actions.index(possible_actions[poss_index])
                    possible_indexes.append(poss_index)
                    action_values.append(self.Q[index,pos_index])


                action_index = np.argmax(action_values)
                action = possible_actions[action_index]

        return action
    
    # perform action
    def perform_action(self,action):
        # update state
        x = self.current_state[0]
        y = self.current_state[1]
        if action == "up":
            y = y-1
        elif action == "down":
            y = y+1
        elif action == "left":
            x = x-1
        elif action == "right":
            x = x+1

        new_state = [x,y]
        self.current_state = [x,y]
        
        # check for reward
        tile = self.map[y][x]
        if tile.isdigit():
            reward = int(tile)
            self.map[y][x] = " "
        else:
            reward = self.move_penalty 
        
        return new_state,reward

    def action_index(self,action):
        action_index = 0
        for actions in self.all_actions:
            if action == actions:
                break
            action_index = action_index+1
        return action_index
    
    def save_Q_table(self,Q,title):
        # write Q table results to textfile
        file_name = title+".txt"
        open(file_name,"w").close()
        file = open(file_name,"a")
        for state in Q:
            index = 0
            for action in state:
                if index < len(state)-1:
                    file.write(str(action)+",")
                else:
                    file.write(str(action))
                index = index+1
            file.write("\n")
        file.close()

    # update state action value matrix
    def update_Q(self,state,new_state,reward,action,new_action):
        x = state[0]
        y = state[1]
        index = y*self.width+x
        action_index = self.action_index(action)

        prediction = self.Q[index,action_index]
        new_x = new_state[0]
        new_y = new_state[1]
        new_index = new_y*self.width+new_x
        new_action_index = self.action_index(new_action)

        target = reward + self.gamma * self.Q[new_index,new_action_index]
        self.Q[index,action_index] = self.Q[index,action_index]+self.learning_rate*(target-prediction)
    def get_raw_Q(self):
        return self.Q
    def set_Q(self, Q):
        self.Q = Q
    def get_Q(self):
        Q_matrix = []
        y = 0
        for row in self.map:
            x = 0
            row_vector = []
            for element in row:
                index = y*self.width+x
                best_action_weight = max(self.Q[index,:])
                if best_action_weight != 0:
                    action_index_tuple = np.where(self.Q[index,:] == best_action_weight)
                    action_index = action_index_tuple[0][0]
                    action = self.all_actions[action_index]
                    row_vector.append(action[0]+" "+str(round(best_action_weight,3)))
                else:
                    row_vector.append("a"+" "+str(round(best_action_weight,3)))
                x = x+1
            Q_matrix.append(row_vector)
            y = y+1
        return Q_matrix

class maze_expected_sarsa:
    def __init__(self):
        # map parameters
        self.map = []
        self.original_map = []
        self.width = 0
        self.height = 0

        # state parameters
        self.current_state = [0,0]
        self.all_actions = []
        self.initial_state = [0,0]

        # algorithm parameters
        self.epsilon = 0 
        self.learning_rate = 0
        self.Q = 0
        self.gamma = 0
        self.move_penalty = 0
        self.total_score = 0

    def select_action(self,state): # epsilon greedy action selecter
        x = state[0]
        y = state[1]

        # determine possible actions
        possible_actions = []
        if self.map[y-1][x] != "#":
            possible_actions.append("up")
        if self.map[y+1][x] != "#":
            possible_actions.append("down")
        if self.map[y][x-1] != "#":
            possible_actions.append("left")
        if self.map[y][x+1] != "#":
            possible_actions.append("right")

        #Epsilon greedy action chooser
        # Check if all action weights are identical
        equal = self.check_identical_weights(x,y)
        index = y*self.width+x
        # If the number is within the epsilon % do random action
        if np.random.uniform(0,1) < self.epsilon or equal == True:
            action = np.random.choice(possible_actions)
        # Else take best action
        else:
            action = ""
            while possible_actions.count(action) == 0:
                # need to take the correct action vector from the width*height possible
                index = y*self.width+x
                # get values of possible actions
                possible_indexes = []
                action_values = []
                for poss_index in range(len(possible_actions)):
                    pos_index = self.all_actions.index(possible_actions[poss_index])
                    possible_indexes.append(poss_index)
                    action_values.append(self.Q[index,pos_index])
                # take best action
                action_index = np.argmax(action_values)
                action = possible_actions[action_index]
        return action
    
    def action_sum(self,state):
        x = state[0]
        y = state[1]
        index = y*self.width+x

        actions_prob = {"up":0, "down":0, "left":0, "right":0}

        # determine possible actions
        possible_actions = []
        if self.map[y-1][x] != "#":
            possible_actions.append(["up",self.Q[index,0]]) # up
        if self.map[y+1][x] != "#":
            possible_actions.append(["down",self.Q[index,1]]) # down
        if self.map[y][x-1] != "#":
            possible_actions.append(["left",self.Q[index,2]]) # left
        if self.map[y][x+1] != "#":
            possible_actions.append(["right",self.Q[index,3]]) # right


        if self.Q[index,0] != 0.0 or self.Q[index,1] != 0.0 or self.Q[index,2] != 0.0 or self.Q[index,3] != 0.0:
            # create list of all Q table action values
            action_values = [self.Q[index,0],self.Q[index,1],self.Q[index,2],self.Q[index,3]]
            # go through all actions
            for action in possible_actions:
                # if best action the probability of taking it is 1-learning_rate
                if action[1] == max(action_values) or len(possible_actions) == 1:
                    actions_prob[action[0]] = 1-self.epsilon
                # else the probability is the learning rate devided with the number of possible actions -1
                else:
                    actions_prob[action[0]] = self.epsilon/(len(possible_actions)-1)
        # If all values are 0 we just choose a random winner
        else:
            Todays_Winner = np.random.choice(possible_actions)
            for action in possible_actions:
                if action == Todays_Winner:
                    actions_prob[action[0]] = 1-self.epsilon
                else:
                    actions_prob[action[0]] = self.epsilon/(len(possible_actions)-1)
        
        # calculate sum
        action_sum = 0
        for action in range(len(self.all_actions)):
            action_sum += actions_prob[self.all_actions[action]]*self.Q[index,action]
        return action_sum
    
    def give_score(self,score):
        self.total_score = self.total_score+score

    def get_score(self):
        return self.total_score
    
    def reset_score(self):
        self.total_score = 0

    def remove_reward(self,state):
        self.map[state[1]][state[0]] = " "

    def set_map(self,maze):
        self.original_map = maze
        self.map = np.copy(self.original_map)
        self.width = len(maze[0])
        self.height = len(maze)

    def load_Q_from_txt(self,filename,actions_num = 4):
        self.Q = np.zeros((self.width*self.height,actions_num))
        file = open(filename,"r")
        lines = file.readlines()
        index = 0
        for line in lines:
            action_index = 0
            for value in line.split(","):
                self.Q[index,action_index] = float(value)
                action_index = action_index+1
            index = index+1

    def reset_map(self):
        self.map = np.copy(self.original_map)
        self.current_state = self.initial_state

    def initialize_parameters(self,epsilon=0.1, move_penalty = -0.1, learning_rate=0.2,gamma=0.9, actions_num = 4,state = [0,0],actions =["up","down","left","right"]):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.current_state = state
        self.initial_state = state
        self.Q = np.zeros((self.width*self.height,actions_num))
        # randomize Qtable with values between 0 and 1
        for position in range(len(self.Q)):
            for action in range(len(self.Q[position])):
                self.Q[position][action] = random.uniform(0,1)
        self.all_actions = actions
        self.move_penalty = move_penalty

    def set_initial_state(self,state):
        self.initial_state = state
        self.current_state = state

    def check_identical_weights(self,x,y):
        index = y*self.width+x
        for weight in self.Q[index,:]:
            if weight != self.Q[index,0]:
                return False
        return True

    # perform action
    def perform_action(self,action):
        # update state
        x = self.current_state[0]
        y = self.current_state[1]
        if action == "up":
            y = y-1
        elif action == "down":
            y = y+1
        elif action == "left":
            x = x-1
        elif action == "right":
            x = x+1

        new_state = [x,y]
        self.current_state = [x,y]
        
        # check for reward
        tile = self.map[y][x]
        if tile.isdigit():
            reward = int(tile)
            self.map[y][x] = " "
        else:
            reward = self.move_penalty 
        
        return new_state,reward
    
    def action_index(self,action):
        action_index = 0
        for actions in self.all_actions:
            if action == actions:
                break
            action_index = action_index+1
        return action_index
    
    def save_Q_table(self,Q,title):
        # write Q table results to textfile
        file_title = title+".txt"
        open(file_title,"w").close()
        file = open(file_title,"a")
        for state in Q:
            index = 0
            for action in state:
                if index < len(state)-1:
                    file.write(str(action)+",")
                else:
                    file.write(str(action))
                index = index+1
            file.write("\n")
        file.close()

    # update state action value matrix
    def update_Q(self,state,new_state,reward,action,new_action,actions_sum):
        x = state[0]
        y = state[1]
        index = y*self.width+x
        action_index = self.action_index(action)

        prediction = self.Q[index,action_index]
        new_x = new_state[0]
        new_y = new_state[1]
        new_index = new_y*self.width+new_x
        new_action_index = self.action_index(new_action)

        target = reward + self.gamma * actions_sum
        self.Q[index,action_index] = self.Q[index,action_index]+self.learning_rate*(target-self.Q[index,action_index])

    def get_raw_Q(self):
        return self.Q
    
    def set_Q(self, Q):
        self.Q = Q

    def get_Q(self):
        Q_matrix = []
        y = 0
        for row in self.map:
            x = 0
            row_vector = []
            for element in row:
                index = y*self.width+x
                best_action_weight = max(self.Q[index,:])
                if best_action_weight != 0:
                    action_index_tuple = np.where(self.Q[index,:] == best_action_weight)
                    action_index = action_index_tuple[0][0]
                    action = self.all_actions[action_index]
                    row_vector.append(action[0]+" "+str(round(best_action_weight,3)))
                else:
                    row_vector.append("a"+" "+str(round(best_action_weight,3)))
                x = x+1
            Q_matrix.append(row_vector)
            y = y+1
        return Q_matrix

class random_mouse:
    def __init__(self,map,initial_state):
        self.state = initial_state
        self.original_map = map
        self.map = np.copy(self.original_map)
        self.sum_reward = 0

    def reset_map(self):
        self.map = np.copy(self.original_map)
    
    def reset_score(self):
        self.sum_reward = 0

    def choose_action(self,state):
        x = state[0]
        y = state[1]

        # determine possible actions
        possible_actions = []
        if self.map[y-1][x] != "#":
            possible_actions.append("up")
        if self.map[y+1][x] != "#":
            possible_actions.append("down")
        if self.map[y][x-1] != "#":
            possible_actions.append("left")
        if self.map[y][x+1] != "#":
            possible_actions.append("right")

        action = random.choice(possible_actions)
        return action
    
    def remove_reward(self,state):
        self.map[state[1]][state[0]] = " "

    def take_action(self,action,state):
        x = state[0]
        y = state[1]
        if action == "up":
            y = y-1
        elif action == "down":
            y = y+1
        elif action == "left":
            x = x-1
        elif action == "right":
            x = x+1
        new_state = [x,y]
        self.state = new_state
        tile = self.map[y][x]
        reward = 0
        if tile.isdigit():
            reward = int(tile)
            self.map[y][x] = " "

        return new_state,reward
    
    def give_reward(self,reward):
        self.sum_reward = self.sum_reward+reward
    
    def get_score(self):
        return self.sum_reward

class illustrator:

    def __init__(self):
        self.image = 0
        self.original_image = 0
        self.layer_image = 0
        self.image_width = 0
        self.image_height = 0
        self.cell_width = 0
        self.cell_height = 0
        self.wall_colour = (0,0,0)
        self.empty_colour = (255,255,255)
        self.cheese_colour = (0,255,255)
        self.spawn_colour = (0,255,0)
        self.player_size = 10
        self.map = map

    def create_image(self, width, height):
        self.image_width = width
        self.image_height = height
        self.image = np.zeros(shape=(width,height,3),dtype=np.uint8)
        self.image.fill(255)

    def initialize_board(self,map):
        self.map = map
        rows = len(map)
        cols = len(map[0])
        # split image into cells depending on map size
        self.cell_width = self.image_width/cols
        self.cell_height = self.image_height/rows

        # draw walls
        y = 0
        for row in map:
            x = 0
            for symbol in row:
                cell_start_x = int(x*self.cell_width)
                cell_start_y = int(y*self.cell_height)
                cell_end_x = int((x+1)*self.cell_width-1)
                cell_end_y = int((y+1)*self.cell_height-1)
                if symbol == "#":
                    self.image = cv2.rectangle(self.image,(cell_start_x,cell_start_y),(cell_end_x,cell_end_y),self.wall_colour,-1)
                if symbol.isdigit():
                    self.image = cv2.rectangle(self.image,(cell_start_x,cell_start_y),(cell_end_x,cell_end_y),self.cheese_colour,-1)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1
                    text_colour = self.wall_colour
                    thickness = 2
                    text_size = cv2.getTextSize(symbol,font,font_scale,thickness)
                    origin_point = (int((cell_end_x-cell_start_x)/2+cell_start_x-text_size[0][0]/2),int((cell_end_y-cell_start_y)/2+cell_start_y+text_size[0][1]/2))
                    self.image = cv2.putText(self.image,symbol,origin_point,font,font_scale,text_colour,thickness,cv2.LINE_AA)
                if symbol == "S":
                    self.image = cv2.rectangle(self.image,(cell_start_x,cell_start_y),(cell_end_x,cell_end_y),self.spawn_colour,-1)
                x = x+1
            y = y+1
        self.original_image = np.copy(self.image)
        self.layer_image = self.image.copy()
    
    def reset_board(self):
        self.image = np.copy(self.original_image)
        self.layer_image = np.copy(self.original_image)

    def update_board(self,x,y,colour,form = "c"):
        self.layer_image = self.image.copy()
        cell_start_x = int(x*self.cell_width)
        cell_start_y = int(y*self.cell_height)
        cell_end_x = int((x+1)*self.cell_width-1)
        cell_end_y = int((y+1)*self.cell_height-1)
        if form == "s":
            self.layer_image = cv2.rectangle(self.image,(cell_start_x,cell_start_y),(cell_end_x,cell_end_y),self.spawn_colour,-1)
        else:
            center_point = (int((cell_end_x-cell_start_x)/2+cell_start_x),int((cell_end_y-cell_start_y)/2+cell_start_y))
            self.layer_image = cv2.circle(self.layer_image,center_point,self.player_size,colour,-1)

    def add_mouse(self,x,y,colour):
        cell_start_x = int(x*self.cell_width)
        cell_start_y = int(y*self.cell_height)
        cell_end_x = int((x+1)*self.cell_width-1)
        cell_end_y = int((y+1)*self.cell_height-1)
        center_point = (int((cell_end_x-cell_start_x)/2+cell_start_x),int((cell_end_y-cell_start_y)/2+cell_start_y))
        self.layer_image = cv2.circle(self.layer_image,center_point,self.player_size,colour,-1)

    def show_Q(self,Q_matrix):
        Q_image = self.image.copy()
        # Remove cheese numbers
        y = 0
        for row in self.map:
            x = 0
            for symbol in row:
                cell_start_x = int(x*self.cell_width)
                cell_start_y = int(y*self.cell_height)
                cell_end_x = int((x+1)*self.cell_width-1)
                cell_end_y = int((y+1)*self.cell_height-1)
                if symbol.isdigit():
                    Q_image= cv2.rectangle(self.image,(cell_start_x,cell_start_y),(cell_end_x,cell_end_y),self.cheese_colour,-1)

                x = x+1
            y = y+1

        y = 0
        for row in Q_matrix:
            x = 0
            for text in row:
                cell_start_x = int(x*self.cell_width)
                cell_start_y = int(y*self.cell_height)
                cell_end_x = int((x+1)*self.cell_width-1)
                cell_end_y = int((y+1)*self.cell_height-1)
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.4
                text_colour = self.wall_colour
                thickness = 1
                text_size = cv2.getTextSize(text,font,font_scale,thickness)
                origin_point = (int((cell_end_x-cell_start_x)/2+cell_start_x-text_size[0][0]/2),int((cell_end_y-cell_start_y)/2+cell_start_y+text_size[0][1]/2))
                Q_image = cv2.putText(Q_image,text,origin_point,font,font_scale,text_colour,thickness,cv2.LINE_AA)
                x = x+1
            y = y+1
        cv2.imshow("Q episode results",Q_image)
        cv2.imwrite("Q_illustration.jpg",Q_image)
        #cv2.waitKey(0)


    def show_board(self):
        cv2.imshow("Current map",self.layer_image)
        cv2.imwrite("Illustrated_map.jpg",self.layer_image)
    
    def remove_goal(self,x,y):
        cell_start_x = int(x*self.cell_width)
        cell_start_y = int(y*self.cell_height)
        cell_end_x = int((x+1)*self.cell_width-1)
        cell_end_y = int((y+1)*self.cell_height-1)
        self.image = cv2.rectangle(self.image,(cell_start_x,cell_start_y),(cell_end_x,cell_end_y),self.empty_colour,-1)
    def get_image(self):
        return self.layer_image

class maze_neural_network:
    def __init__(self,layers,map,initial_weights,biases,learning_rate):
        self.weights = initial_weights # list of weights for every layer
        self.map = map # state map
        self.layers = layers # number of layers
        self.biases = biases # list of bias for each layer
        self.input_size = len(self.weights[0]) # number of inputs
        self.layer_sizes = [] # number of perceptrons in each layer
        self.current_layer_outputs = [0]*layers # current activations
        index = 0
        self.training_percentage = 1
        self.input = []
        self.deltas = []
        self.final_output_size = 2
        self.learning_rate = learning_rate
        while index < len(self.weights):
            self.layer_sizes.append(len(self.weights[index]))
            index = index + 1
    
    def randomize_weights(self):
        for layer in range(self.layers):
            for perceptron in range(self.layer_sizes[layer]):
                for weight in range(len(self.weights[layer][perceptron])):
                    self.weights[layer][perceptron][weight] = random.uniform(0,1)
    
    # different activation functions
    def sigmoid_activation(self,x):
        # Smooth s shape between 0 and 1
        y = 1.0/(1.0+np.exp(-x))
        return y
    
    def rectified_linear_unit_activation(self,x):
        # gives positive value or 0 if negative value
        result = []
        for i in range(len(x)):
            value = x[i]
            result.append(max(0,value))
        return np.array(result)
    
    def leaky_rectified_linear_unit_activation(self,x,alpha=0.1):
        # small value for negative values or just the positive value
        result = []
        for i in range(len(x)):
            value = x[i]
            result.append(max(alpha*value,value))
        return np.array(result)
    
    def tanh_activation(self,x):
        # S shape curve between -1 and 1 centered at 0
        return np.tanh(x)
    
    def softmax_activation(self,x):
        # vector of real values into probability distribution over classes
        exponents = np.exp(x)
        y = np.divide(exponents,np.sum(exponents))
        return y
    
    def layer(self,layer_num, inputs, activation_type = 0):
        # read current weights and biase
        weights = np.array(self.weights[layer_num]) # number equal to inputs
        input_array = np.array(inputs)
        bias = np.array(self.biases[layer_num])
        input_array = np.append(input_array,bias)
        # predict
        try:
            activations = []
            for weight_array in weights:
                activations.append(np.dot(input_array,weight_array))
        except:
            print("Could not calculate prediction")
        # apply activation if last layer
        outputs = np.array(activations)
        if layer_num > -1:
            if activation_type == 0:
                outputs = self.sigmoid_activation(np.array(activations, dtype=np.float128))
            elif activation_type == 1:
                outputs = self.rectified_linear_unit_activation(np.array(activations, dtype=np.float128))
            elif activation_type == 2:
                outputs= self.leaky_rectified_linear_unit_activation(np.array(activations, dtype=np.float128))
            elif activation_type == 3:
                outputs = self.tanh_activation(np.array(activations, dtype=np.float128))
            elif activation_type == 4:
                outputs = self.softmax_activation(np.array(activations, dtype=np.float128))

        self.current_layer_outputs[layer_num] = outputs
        return outputs
    
    def forward_pass(self,input_values):
        # input = [x,y,cheese_up,cheese_down, cheese_left, cheese_right]
        input = input_values
        self.input = input
        for index in range(self.layers):
            output = self.layer(index,input,0) # 0 -> sigmoid activation function
            input = output
        return output
    
    def mean_square_error(self,result,desired):
        mean_square_error = 1/2*(desired-result)**2
        return mean_square_error
    
    def gradient_mean_square_error(self,result,desired):
        gradient_mean_square_error = (result-desired)
        return gradient_mean_square_error
    
    def sigmoid_gradient(self,output):
        return output*(1.0-output)
    
    def update_weights(self):
        #self.biases = self.biases-(self.bias_error*self.learning_rate)
        #self.weights = self.weights-(self.weight_error*self.learning_rate)

        # go through each layer
        for layer in range(self.layers):
            # go through each perceptron
            for perceptron in range(self.layer_sizes[layer]):
                # update bias weight
                self.weights[layer][perceptron][-1] -=self.learning_rate*self.deltas[layer][perceptron]*self.biases[layer]
                # if first layer
                if layer == 0:
                    # go through all inputs
                    for input in range(len(self.input)):
                        # update weights
                        self.weights[layer][perceptron][input] -= self.learning_rate*self.deltas[layer][perceptron]*input
                else:
                    # go through previous layer outputs
                    for prev_output in range(len(self.current_layer_outputs[layer-1])):
                        # update weights
                        self.weights[layer][perceptron][prev_output] -= self.learning_rate*self.deltas[layer][perceptron]*self.current_layer_outputs[layer-1][prev_output]
         
    def backpropagation(self,desired):
        #prediction = self.current_layer_outputs[self.layers-1]
        #prediction_error = 2*(prediction-desired)
        #layer_2 = self.current_layer_outputs[self.layers-2]
        #prediction_layer2 = self.sigmoid_gradient(layer_2)
        #layer_1 = self.current_layer_outputs[self.layers-1]
        #prediction_layer1 = self.sigmoid_gradient(layer_1)
        #layer_1_bias = 1
        #layer_2_bias = 1
        #layer_2_weights = (0*self.weights[self.layers-1])+(1*self.current_layer_outputs[self.layers-2])
        #layer_1_weights = (0*self.weights[self.layers-1])+(1*self.input)
        #bias_error = (prediction_error*prediction_layer2*layer_2_bias*prediction_layer1*layer_1_bias)
        #error_weight = (prediction_error*prediction_layer2*layer_2_weights*prediction_layer1*layer_1_weights)
        #self.bias_error = bias_error
        #self.weight_error = error_weight
        # initialize delta list
        deltas = []
        ## initialize error list
        errors = []
        ## go trough layers backwards
        for current_layer in reversed(range(self.layers)):
            # initialize layer errors
            layer_errors = []
            # initialize layer delta
            delta = []
        
            # If not output layer
            if current_layer != self.layers-1:
                # go through each perceptron in layer
                for perceptron in range(self.layer_sizes[current_layer]):
                    # Ininialize error
                    error = 0.0
                    # go through each perceptron in next layer
                    for next_perceptron in range(self.layer_sizes[current_layer+1]):
                        # increment error
                        error = error + deltas[((self.layers-1)-current_layer)-1][next_perceptron] #self.weights[current_layer+1][next_perceptron][perceptron]*
                    # append error list
                    layer_errors.append(error)
            # if output layer
            else:
                # go through each perceptron in layer
                for perceptron in range(self.layer_sizes[current_layer]):
                    # get output
                    output = self.current_layer_outputs[current_layer][perceptron]
                    # calculate error
                    error = self.gradient_mean_square_error(output,desired[perceptron])
                    # append error list
                    layer_errors.append(error)
            # go through each perceptron
            for perceptron in range(self.layer_sizes[current_layer]):
                #Epoch 0 sum error: 15.000000 get ouput
                output = self.current_layer_outputs[current_layer][perceptron]
                # calculate delta
                delta.append(layer_errors[perceptron]*self.sigmoid_gradient(output))
            deltas.append(delta)
            errors.append(layer_errors)
        self.deltas = deltas
        self.deltas.reverse()

    def test_network(self,desired,input,iterations):
        # open data file
        open("Output_1_Data.txt","w").close()
        open("Output_2_Data.txt","w").close()
        file1 = open("Output_1_Data.txt","a")
        file2 = open("Output_2_Data.txt","a")
        # create date lists
        o1 = []
        o2 = []
        # randomize weights
        self.randomize_weights()
        # loop through iterations and see how close to desired result we get
        for i in range(iterations):
            # forward pass
            result = self.forward_pass(input)
            if i == 0:
                old_results = result
            # test result
            o1.append(result[0])
            o2.append(result[1])
            # backpropagation
            self.backpropagation(desired)
            # update weights
            self.update_weights()
        
        # write results to file
        for index in range(len(o1)):
            file1.write(str(o1[index]))
            file1.write("\n")
            file2.write(str(o2[index]))
            file2.write("\n")
        file1.close()
        file2.close()
        print("Final result:",result)
        print("Starting output:",old_results)
        print("starting difference:",abs(old_results-desired))
        print("new difference:",abs(result-desired))

    def save_weights(self):
        # write weight table to textfile
        open("neural_network_weights.txt","w").close()
        file = open("neural_network_weights.txt","a")
        for layer in range(self.layers):
            for perceptron in range(self.layer_sizes[layer]):
                index = 0
                for weight in self.weights[layer][perceptron]:
                    if index < len(self.weights[layer][perceptron])-1:
                        file.write(str(weight))
                        file.write(",")
                    else:
                        file.write(str(weight))
                    index = index +1
                file.write("\n")
        file.close()

    def load_weights(self,filename):
        file = open(filename,"r")
        lines = file.readlines()
        weight_index = 0
        perceptron_index = 0
        layer_index = 0
        for line in lines:
            if perceptron_index >= self.layer_sizes[layer_index]:
                perceptron_index = 0
                layer_index += 1
            for weight in line:
                print(weight)
                self.weights[layer_index][perceptron_index][weight_index] = float(weight)
                weight_index += 1
            perceptron_index += 1
        file.close()

    def train_network(self, training_data, epochs):
        # training data should be a list containing list of the form [input,desired]

        # split data set into training and validation
        training_max_index = int(len(training_data)*self.training_percentage)-2
        last_error_sum = 1000
        # run through all epochs
        for epoch in range(epochs):
            error_sum = 0
            index = 0
            # go through training set
            for data in training_data:
                # get variable data
                input = training_data[index][0]
                desired = training_data[index][1]
                # forward pass
                result = self.forward_pass(input)
                # train if in training set
                if index <= training_max_index:
                    # backpropagation
                    self.backpropagation(desired)
                    # update weights
                    self.update_weights()
                # else validate
                else:
                    error_sum += sum(abs(result-desired))
                index += 1

            print("Epoch %i sum error: %f" %(epoch,error_sum))
            if error_sum > last_error_sum+0.01:
                print("Error incresed. Stopping")
                break
            last_error_sum = error_sum
    
    def get_training_data_from_file(self,filename):
        file = open(filename,'r')
        lines = file.readlines()
        training_data = []
        # remove first line
        lines.pop(0)
        for line in lines:
            if len(line) > 0:
                data = []
                desired = []
                string_data = line.split(",")
                for i in range(len(string_data)):
                    if i < len(string_data)-self.final_output_size:
                        data.append(int(string_data[i]))
                    else:
                        element = string_data[i].replace("\n","")
                        desired.append(int(element))
                combined = [data,desired]
                training_data.append(combined)
        return training_data

    def convert_output_to_maze_action(self,output):
        action = ""
        if(int(output[0]) == 0 and int(output[1]) == 0):
            action = "up"
        elif(int(output[0]) == 0 and int(output[1]) == 1):
            action = "down"
        elif(int(output[0]) == 1 and int(output[1]) == 0):
            action = "left"
        elif(int(output[0]) == 1 and int(output[1]) == 1):
            action == "right"
        return action

class experiment:
    def __init__(self):
        self.map = []
        self.original_map =[]
        self.episodes = 0
        self.initial_state = (0,0)

    def set_parameters(self,map,episodes):
        self.map = np.copy(map)
        self.original_map = np.copy(map)
        self.episodes = episodes

        # find start position
        y = 0
        for row in map:
            x = 0
            for value in row:
                if value == "S":
                    self.initial_state = (x,y)
                    return
                x = x+1
            y = y+1
        print("No initial state found in maze")

    def convert_output_to_maze_action(self,output):
        action = ""
        if(int(output[0]) == 0 and int(output[1]) == 0):
            action = "up"
        elif(int(output[0]) == 0 and int(output[1]) == 1):
            action = "down"
        elif(int(output[0]) == 1 and int(output[1]) == 0):
            action = "left"
        elif(int(output[0]) == 1 and int(output[1]) == 1):
            action == "right"
        return action

    def remaining_cheese(self):
        count = 0
        for row in self.map:
            for symbol in row:
                if symbol.isdigit():
                    count = count+1
        return count
    
    def insert_random_start(self):
        rows = len(self.map)
        cols = len(self.map[0])
        row = random.randrange(0,rows)
        col = random.randrange(0,cols)
        while self.map[row][col] == '#' or self.map[row][col].isdigit():
            row = random.randrange(0,rows-1)
            col = random.randrange(0,cols-1)
        self.initial_state = (col,row)

    def train_sarsa(self, epsilon,penalty,learning_rate,gamma):
        # initialize classes
        sarsa = maze_sarsa()

        # give classes needed parameters
        sarsa.set_map(np.copy(self.map))
        sarsa.initialize_parameters(epsilon=epsilon, move_penalty=penalty,learning_rate=learning_rate,gamma=gamma,state=self.initial_state)

        # set up visualizer
        #visualize = illustrator()
        #visualize.create_image(660,660)
        #visualize.initialize_board(np.copy(self.map))

        # setup step counter list
        episode_steps = []

        # Run through experiment n times
        for episode in range(self.episodes):
            index = 0
            # Set random initial start
            # Create random start
            self.insert_random_start()
            #visualize.update_board(self.initial_state[0],self.initial_state[1],(0,255,0),"s")
            sarsa.set_initial_state(self.initial_state)
            
            # get start state and choose start action
            state = self.initial_state
            action = sarsa.select_action(state)

            cheese_count = self.remaining_cheese()
            max_cheese = cheese_count
            reward = 0
            step_count = 0
            # show map
            print("Episode: %i" %episode)
            #visualize.update_board(state[0],state[1],(0,0,255))
            #visualize.show_board()

            # make steps until one cheese have been taken
            while cheese_count == max_cheese:
                # Check if within wall
                if self.map[state[1]][state[0]] == "#":
                    print("ERROR: Stuck in wall")
                    #visualize.show_board()
                    cv2.waitKey(0)

                # take action
                new_state,reward = sarsa.perform_action(action)
                step_count = step_count+1
                # show map
                #visualize.update_board(new_state[0],new_state[1],(0,0,255))   
                #visualize.show_board()
                if reward > 0:
                    print("reward")
                    #visualize.remove_goal(new_state[0],new_state[1])

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                # choose next action
                new_action = sarsa.select_action(new_state)
                # Clear old spot
                x = state[0]
                y = state[1]
                self.map[y][x] = " "

                # Update Q
                sarsa.update_Q(state,new_state,reward,action,new_action)

                # update state, action, index and map
                state = new_state
                action = new_action
                index = index+1
                x = state[0]
                y = state[1]
                self.map[y][x] = "P"
                cheese_count = self.remaining_cheese()
                # test stuff
                #if cheese_count < 10:
                #    cheese_count = 0

            # Episode sum up
            Q_results = sarsa.get_Q()
            #visualize.show_Q(Q_results)
            print("Total steps taken %i" %step_count)
            self.map = np.copy(self.original_map)
            episode_steps.append(step_count)
            sarsa.reset_map()
            #visualize.reset_board()

        # print step improvement

        return sarsa.get_raw_Q(),episode_steps
    
    def train_expected_sarsa(self,epsilon,penalty,learning_rate,gamma):
        # initialize classes
        expected_sarsa = maze_expected_sarsa()

        # give classes needed parameters
        expected_sarsa.set_map(np.copy(self.map))
        expected_sarsa.initialize_parameters(epsilon=epsilon, move_penalty=penalty,learning_rate=learning_rate,gamma=gamma,state=self.initial_state)

        # set up visualizer
        visualize = illustrator()
        #visualize.create_image(660,660)
        #visualize.initialize_board(np.copy(self.map))

        # setup step counter list
        episode_steps = []

        # Run through experiment n times
        for episode in range(self.episodes):
            index = 0
            # Set random initial start
            # Create random start
            self.insert_random_start()
            #visualize.update_board(self.initial_state[0],self.initial_state[1],(0,255,0),"s")
            expected_sarsa.set_initial_state(self.initial_state)
            
            # get start state and choose start action
            state = self.initial_state
            action = expected_sarsa.select_action(state)

            cheese_count = self.remaining_cheese()
            max_cheese = cheese_count
            reward = 0
            step_count = 0
            # show map
            print("Episode: %i" %episode)
            #visualize.update_board(state[0],state[1],(0,0,255))
            #visualize.show_board()

            # make steps until one cheese have been taken
            while cheese_count == max_cheese:
                # Check if within wall
                if self.map[state[1]][state[0]] == "#":
                    print("ERROR: Stuck in wall")
                    #visualize.show_board()
                    cv2.waitKey(0)

                # take action
                new_state,reward = expected_sarsa.perform_action(action)
                step_count = step_count+1
                # show map
                #visualize.update_board(new_state[0],new_state[1],(0,0,255))   
                #visualize.show_board()
                #if reward > 0:
                #    visualize.remove_goal(new_state[0],new_state[1])

                #k = cv2.waitKey(30) & 0xff
                #if k == 27:
                #    break

                # choose next action
                new_action = expected_sarsa.select_action(new_state)
                # Clear old spot
                x = state[0]
                y = state[1]
                self.map[y][x] = " "

                # get action probabilities sum
                actions_sum = expected_sarsa.action_sum(new_state)

                # Update Q
                expected_sarsa.update_Q(state,new_state,reward,action,new_action,actions_sum)

                # update state, action, index and map
                state = new_state
                action = new_action
                index = index+1
                x = state[0]
                y = state[1]
                self.map[y][x] = "P"
                cheese_count = self.remaining_cheese()

            # Episode sum up
            Q_results = expected_sarsa.get_Q()
            #visualize.show_Q(Q_results)
            print("Total steps taken %i" %step_count)
            self.map = np.copy(self.original_map)
            episode_steps.append(step_count)
            expected_sarsa.reset_map()
            #visualize.reset_board()

        # print step improvement
        print(episode_steps)
        return expected_sarsa.get_raw_Q()

    def find_closest_cheese(self,state):
        cheeses_found = []
        distance = 0
        # expand around state until cheese is encountered
        while len(cheeses_found) == 0:
            distance = distance+1
            # get row and column limits
            min_row = state[0]-distance
            max_row = state[0]+distance
            min_col = state[1]-distance
            max_col = state[1]+distance
            if min_row < 0:
                min_row = 0
            if max_row >= len(self.map):
                max_row = len(self.map)-1
            if min_col < 0:
                min_col = 0
            if max_col >= len(self.map[0]):
                max_col = len(self.map[0])-1
            # go through square around state
            for row in range(min_row,max_row+1,1):
                for col in range(min_col,max_col+1,1):
                    # ignore values where row or col is not at a limit
                    if row == min_row or row == max_row or col == min_col or col == max_col:
                        # check if cheese
                        tile = self.map[row][col]
                        if tile.isdigit():
                            reward = int(tile)
                            cheeses_found.append([reward,row,col])
        # With cheese candidates found we choose the one with the biggest reward
        best_cheese = [0,0,0]
        for cheese in cheeses_found:
            if cheese[0] > best_cheese[0]:
                best_cheese = cheese
        # determine direction
        if best_cheese[1] == state[0]: # row is the same then its horizontal
            if best_cheese[2] > state[1]:
                # right
                direction = 3
            else:
                # left
                direction = 2
        elif best_cheese[2] == state[1]: # if col is the same then its vertical
            if best_cheese[1] > state[0]:
                #down
                direction = 1
            else:
                #up
                direction = 0
        else: # same distance so in a corner
            # choose random of two possible directions
            if best_cheese[1] > state[0] and best_cheese[2] > state[1]: # bottom right
                direction = 4
            elif best_cheese[1] > state[0] and best_cheese[2] < state[1]: # bottom left
                direction = 5
            elif best_cheese[1] < state[0] and best_cheese[2] > state[1]: # top right
                direction = 6
            elif best_cheese[1] < state[0] and best_cheese[2] < state[1]: # top left
                direction = 7
            else: # error
                direction = -1
        return direction,distance
    
    def find_visible_cheese(self,state):
        cheeses_found = []
        distance = 0
        # loop left, up, down and left until cheese or wall reached
        left_done = False
        right_done = False
        up_done = False
        down_done = False
        while len(cheeses_found) == 0 and distance < len(self.map):
            distance = distance + 1
            # left
            if state[1]-distance >= 0 and left_done == False:
                item = self.map[state[0],state[1]-distance]
                if item.isdigit():
                    cheeses_found.append([int(item),2,distance])
                elif item == "#":
                    left_done = True
            # right
            if state[1]+distance < len(self.map[0]) and right_done == False:
                item = self.map[state[0],state[1]+distance]
                if item.isdigit():
                    cheeses_found.append([int(item),3,distance])
                elif item == "#":
                    right_done = True
            # up
            if state[0]-distance >= 0 and up_done == False:
                item = self.map[state[0]-distance,state[1]]
                if item.isdigit():
                    cheeses_found.append([int(item),0,distance])
                elif item == "#":
                    up_done = True
            # down
            if state[0]+distance < len(self.map) and down_done == False:
                item = self.map[state[0]+distance,state[1]]
                if item.isdigit():
                    cheeses_found.append([int(item),1,distance])
                elif item == "#":
                    down_done = True
        # take best cheese
        best_cheese = [0,0,0]
        if len(cheeses_found) == 0:
            direction = -1
            distance = -1
        else:
            for cheese in cheeses_found:
                if cheese[0] > best_cheese[0]:
                    best_cheese = cheese
            direction = best_cheese[1]
            distance = best_cheese[2]
        return direction,distance

    def check_action(self,action,state,action_num):
        last_state = state
        new_state = [0,0]
        reward = 0
        if action == "up":
            new_state[1] = last_state[1]
            new_state[0] = last_state[0]-1
        elif action == "down":
            new_state[1] = last_state[1]
            new_state[0] = last_state[0]+1
        elif action == "left":
            new_state[1] = last_state[1]-1
            new_state[0] = last_state[0]
        elif action == "right":
            new_state[1] = last_state[1]+1
            new_state[0] = last_state[0]
        index = 0
        new_action_num = action_num
        while new_state[0] > len(self.map) or new_state[0] < 0 or new_state[1] > len(self.map[0]) or new_state[1] < 0 or self.map[new_state[0]][new_state[1]] == "#":
            last_index = 0
            if new_action_num[0] > new_action_num[1] and new_action_num[0] != 1:
                print("1")
                new_action_num[0] = 1
                last_index = 0
            elif new_action_num[1] > new_action_num[0] and new_action_num[1] != 1:
                print("2")
                new_action_num[1] = 1
                last_index = 1
            elif new_action_num[1] == 1 and new_action_num[0] == 0:
                print("3")
                new_action_num[0] = 1
                last_index = 0
            elif new_action_num[0] == 1 and new_action_num[1] == 0:
                print("4")
                new_action_num[1] = 1
                last_index = 1
            elif new_action_num[last_index] == 1:
                print("5")
                new_action_num[0] = 0 
            elif new_action_num[last_index] == 0:
                print("6")
                new_action_num[1] = 0
            new_action = self.convert_output_to_maze_action(new_action_num)

            if new_action == "up":
                new_state[1] = last_state[1]-1
                new_state[0] = last_state[0]
            elif new_action == "down":
                new_state[1] = last_state[1]+1
                new_state[0] = last_state[0]
            elif new_action == "left":
                new_state[1] = last_state[1]
                new_state[0] = last_state[0]-1
            elif new_action == "right":
                new_state[1] = last_state[1]
                new_state[0] = last_state[0]+1
            if index > 3:
                print("Could not find a solution")
                new_state = state
            index +=1

        if self.map[new_state[0]][new_state[1]].isdigit():
            reward = int(self.map[new_state[0]][new_state[1]])
        return new_state,reward

    def determine_direction_value(self,direction):
        if direction == '#':
            result = -1
        elif direction.isdigit():
            result = int(direction)
        else:
            result = 0
        return result

    def get_training_data_neural_network(self,title):
        # create test file.
        file_name = title+".txt"
        open(file_name,"w").close()
        file = open(file_name,"a")
        file.write("eye direction,eye distance,nose direction,nose distance,up,down,left,right,desired_1,desired_2\n")
        # go through entire map
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                # if cheese or wall ignore else create test set data
                if self.map[row][col] != '#' and self.map[row][col].isdigit() == False:
                    print(row,col)
                    # find direction and distance
                    nose_direction,nose_distance = self.find_closest_cheese([row,col])
                    eye_direction,eye_distance = self.find_visible_cheese([row,col])
                    # get element in all four directions
                    if row-1 >= 0:
                        up = self.map[row-1][col]
                        up = self.determine_direction_value(up)
                    else:
                        up = -1

                    if row +1 <= len(self.map):
                        down = self.map[row+1][col]
                        down = self.determine_direction_value(down)
                    else:
                        down = -1

                    if col -1 >= 0:
                        left = self.map[row][col-1]
                        left = self.determine_direction_value(left)
                    else:
                        left = -1

                    if col +1 <= len(self.map[0]):
                        right = self.map[row][col+1]
                        right = self.determine_direction_value(right)
                    else:
                        right = -1

                    # write data
                    file.write(str(eye_direction)+","+str(eye_distance)+","+str(nose_direction) + "," + str(nose_distance) +"," + str(up) + "," + str(down) + "," + str(left) + "," + str(right) + "," + ",\n" )
        file.close()

    def sarsa_vs_random(self,epsilon,learning_rate,gamma, Q_file = 0,load = 1,):
        # initialize classes
        sarsa = maze_sarsa()
        rando = random_mouse(np.copy(self.original_map),self.initial_state)

        # give classes needed parameters
        sarsa.set_map(np.copy(self.map))
        sarsa.initialize_parameters(epsilon=epsilon,learning_rate=learning_rate,gamma=gamma, state=self.initial_state,move_penalty=-1)
        if load == 1:
            sarsa.load_Q_from_txt(Q_file)

        # set up visualizer
        #visualize = illustrator()
        #visualize.create_image(660,660)
        #visualize.initialize_board(np.copy(self.map))

        # setup stats list
        episode_steps = []
        rando_scores = []
        sarsa_scores = []
        winners = []

        # stop sarsa backtracking with punishment
        punishment = -10
        two_states_ago = 0

        # Run through experiment n times
        for episode in range(self.episodes):
            index = 0
            
            # get start state and choose start action for each mouse
            sarsa_state = self.initial_state
            rando_state = self.initial_state
            sarsa_action = sarsa.select_action(sarsa_state)
            rando_action = rando.choose_action(rando_state)


            cheese_count = self.remaining_cheese()
            sarsa_reward = 0
            rando_reward = 0
            step_count = 0
            # show map
            print("Episode: %i" %episode)
            #if sarsa_state == rando_state:
                #visualize.update_board(sarsa_state[0],sarsa_state[1],(255,0,255))
            #else:
                #visualize.update_board(rando_state[0],rando_state[1],(255,0,0))
                #visualize.add_mouse(sarsa_state[0],sarsa_state[1],(0,0,255))
            #visualize.show_board()
        
            # make steps until all cheese has been taken
            while cheese_count > 0:
                # Check if within wall
                if self.map[sarsa_state[1]][sarsa_state[0]] == "#":
                    print("ERROR: Sarsa stuck in wall")
                    #visualize.show_board()
                    cv2.waitKey(0)
                elif self.map[rando_state[1]][rando_state[0]] == "#":
                    print("ERROR: Rando stuck in wall")
                    #visualize.show_board()
                    cv2.waitKey(0)

                # take action
                new_sarsa_state,sarsa_reward = sarsa.perform_action(sarsa_action)
                new_rando_state,rando_reward = rando.take_action(rando_action,rando_state)
                step_count = step_count+1
                # show map
                #if new_sarsa_state == new_rando_state:
                    #visualize.update_board(new_sarsa_state[0],new_sarsa_state[1],(255,0,255))
                #else:
                #    visualize.update_board(new_rando_state[0],new_rando_state[1],(255,0,0))
                #    visualize.add_mouse(new_sarsa_state[0],new_sarsa_state[1],(0,0,255))
                #visualize.show_board()

                # Reward handling
                if new_sarsa_state == two_states_ago:
                    sarsa_reward = punishment
                if sarsa_reward > 0 or rando_reward > 0:
                    if new_sarsa_state == new_rando_state:
                        sarsa_reward = sarsa_reward/2
                        rando_reward = rando_reward/2
                        rando.give_reward(rando_reward)
                        sarsa.give_score(sarsa_reward)
                        #visualize.remove_goal(new_sarsa_state[0],new_sarsa_state[1])
                    else:
                        if sarsa_reward > 0:
                            rando.remove_reward(new_sarsa_state)
                            sarsa.give_score(sarsa_reward)
                            #visualize.remove_goal(new_sarsa_state[0],new_sarsa_state[1])
                        if rando_reward > 0:
                            sarsa.remove_reward(new_rando_state)
                            rando.give_reward(rando_reward)
                            #visualize.remove_goal(new_rando_state[0],new_rando_state[1])

                #k = cv2.waitKey(30) & 0xff
                #if k == 27:
                #    break

                # choose next action
                new_sarsa_action = sarsa.select_action(new_sarsa_state)
                new_rando_action = rando.choose_action(new_rando_state)
                # Clear old spot
                self.map[sarsa_state[1]][sarsa_state[0]] = " "
                self.map[rando_state[1]][rando_state[0]] = " "

                # Update Q
                sarsa.update_Q(sarsa_state,new_sarsa_state,sarsa_reward,sarsa_action,new_sarsa_action)

                # update state, action, index and map
                two_states_ago = sarsa_state
                sarsa_state = new_sarsa_state
                sarsa_action = new_sarsa_action
                rando_state = new_rando_state
                rando_action = new_rando_action
                index = index+1
                self.map[sarsa_state[1]][sarsa_state[0]] = "s"
                self.map[rando_state[1]][rando_state[0]] = "r"
                cheese_count = self.remaining_cheese()

            # Post test results
            sarsa_score = sarsa.get_score()
            rando_score = rando.get_score()
            print("Sarsa result: %f cheese" %sarsa_score)
            print("Rando result: %f cheese" %rando_score)
            print("Total steps taken %i" %step_count)
            self.map = np.copy(self.original_map)
            episode_steps.append(step_count)
            sarsa.reset_map()
            rando.reset_map()
            #visualize.reset_board()
            sarsa.reset_score()
            rando.reset_score()
            if load == 1:
                sarsa.load_Q_from_txt(Q_file)
            rando_scores.append(rando_score)
            sarsa_scores.append(sarsa_score)
            if rando_score == sarsa_score:
                print("Its a tie!")
                winners.append("tie")
            elif rando_score > sarsa_score:
                print("Random mouse wins!")
                winners.append("random")
            else:
                print("Sarsa mouse wins!")
                winners.append("sarsa")
        
        return winners,rando_scores,sarsa_scores,episode_steps

    def expected_sarsa_vs_random(self, epsilon,learning_rate,gamma, Q_file = 0,load = 1,):
        # initialize classes
        expected_sarsa = maze_expected_sarsa()
        rando = random_mouse(np.copy(self.original_map),self.initial_state)

        # give classes needed parameters
        expected_sarsa.set_map(np.copy(self.map))
        expected_sarsa.initialize_parameters(epsilon=epsilon,learning_rate=learning_rate,gamma=gamma, state=self.initial_state,move_penalty=-1)
        if load == 1:
            expected_sarsa.load_Q_from_txt(Q_file)

        # set up visualizer
        #visualize = illustrator()
        #isualize.create_image(660,660)
        #visualize.initialize_board(np.copy(self.map))

        # setup stats list
        episode_steps = []
        rando_scores = []
        expected_sarsa_scores = []
        winners = []

        # stop expected sarsa backtracking with punishment
        punishment = -10
        two_states_ago = 0

        # Run through experiment n times
        for episode in range(self.episodes):
            index = 0
            
            # get start state and choose start action for each mouse
            expected_sarsa_state = self.initial_state
            rando_state = self.initial_state
            expected_sarsa_action = expected_sarsa.select_action(expected_sarsa_state)
            rando_action = rando.choose_action(rando_state)


            cheese_count = self.remaining_cheese()
            expected_sarsa_reward = 0
            rando_reward = 0
            step_count = 0
            # show map
            print("Episode: %i" %episode)
            #if expected_sarsa_state == rando_state:
            #    visualize.update_board(expected_sarsa_state[0],expected_sarsa_state[1],(255,0,100))
            #else:
            #    visualize.update_board(rando_state[0],rando_state[1],(255,0,0))
            #    visualize.add_mouse(expected_sarsa_state[0],expected_sarsa_state[1],(0,0,100))
            #visualize.show_board()
        
            # make steps until all cheese has been taken
            while cheese_count > 0:
                # Check if within wall
                if self.map[expected_sarsa_state[1]][expected_sarsa_state[0]] == "#":
                    print("ERROR: Expected sarsa stuck in wall")
                    #visualize.show_board()
                    #cv2.waitKey(0)
                elif self.map[rando_state[1]][rando_state[0]] == "#":
                    print("ERROR: Rando stuck in wall")
                    #visualize.show_board()
                    #cv2.waitKey(0)

                # take action
                new_expected_sarsa_state,expected_sarsa_reward = expected_sarsa.perform_action(expected_sarsa_action)
                new_rando_state,rando_reward = rando.take_action(rando_action,rando_state)
                step_count = step_count+1
                # show map
                #if new_expected_sarsa_state == new_rando_state:
                #    visualize.update_board(new_expected_sarsa_state[0],new_expected_sarsa_state[1],(255,0,100))
                #else:
                #    visualize.update_board(new_rando_state[0],new_rando_state[1],(255,0,0))
                #    visualize.add_mouse(new_expected_sarsa_state[0],new_expected_sarsa_state[1],(0,0,100))
                #visualize.show_board()

                # Reward handling
                if new_expected_sarsa_state == two_states_ago:
                    expected_sarsa_reward = punishment
                if expected_sarsa_reward > 0 or rando_reward > 0:
                    if new_expected_sarsa_state == new_rando_state:
                        expected_sarsa_reward = expected_sarsa_reward/2
                        rando_reward = rando_reward/2
                        rando.give_reward(rando_reward)
                        expected_sarsa.give_score(expected_sarsa_reward)
                        #visualize.remove_goal(new_expected_sarsa_state[0],new_expected_sarsa_state[1])
                    else:
                        if expected_sarsa_reward > 0:
                            rando.remove_reward(new_expected_sarsa_state)
                            expected_sarsa.give_score(expected_sarsa_reward)
                            #visualize.remove_goal(new_expected_sarsa_state[0],new_expected_sarsa_state[1])
                        if rando_reward > 0:
                            expected_sarsa.remove_reward(new_rando_state)
                            rando.give_reward(rando_reward)
                            #visualize.remove_goal(new_rando_state[0],new_rando_state[1])

                #k = cv2.waitKey(30) & 0xff
                #"if k == 27:
                #"    break

                # choose next action
                new_expected_sarsa_action = expected_sarsa.select_action(new_expected_sarsa_state)
                new_rando_action = rando.choose_action(new_rando_state)
                # Clear old spot
                self.map[expected_sarsa_state[1]][expected_sarsa_state[0]] = " "
                self.map[rando_state[1]][rando_state[0]] = " "

                # get action probabilities sum
                actions_sum = expected_sarsa.action_sum(new_expected_sarsa_state)
                # Update Q
                expected_sarsa.update_Q(expected_sarsa_state,new_expected_sarsa_state,expected_sarsa_reward,expected_sarsa_action,new_expected_sarsa_action,actions_sum)

                # update state, action, index and map
                two_states_ago = expected_sarsa_state
                expected_sarsa_state = new_expected_sarsa_state
                expected_sarsa_action = new_expected_sarsa_action
                rando_state = new_rando_state
                rando_action = new_rando_action
                index = index+1
                self.map[expected_sarsa_state[1]][expected_sarsa_state[0]] = "s"
                self.map[rando_state[1]][rando_state[0]] = "r"
                cheese_count = self.remaining_cheese()

            # Post test results
            expected_sarsa_score = expected_sarsa.get_score()
            rando_score = rando.get_score()
            print("Expected sarsa result: %f cheese" %expected_sarsa_score)
            print("Rando result: %f cheese" %rando_score)
            print("Total steps taken %i" %step_count)
            self.map = np.copy(self.original_map)
            episode_steps.append(step_count)
            expected_sarsa.reset_map()
            rando.reset_map()
            #visualize.reset_board()
            expected_sarsa.reset_score()
            rando.reset_score()
            if load == 1:
                expected_sarsa.load_Q_from_txt(Q_file)
            rando_scores.append(rando_score)
            expected_sarsa_scores.append(expected_sarsa_score)
            if rando_score == expected_sarsa_score:
                print("Its a tie!")
                winners.append("tie")
            elif rando_score > expected_sarsa_score:
                print("Random mouse wins!")
                winners.append("random")
            else:
                print("Expected sarsa mouse wins!")
                winners.append("expected sarsa")
        
        return winners,rando_scores,expected_sarsa_scores,episode_steps

    def ANN_vs_random(self, Q_file = 0,load = 1,):
        # initialize classes
        rando = random_mouse(np.copy(self.original_map),self.initial_state)
        layers = 4 # 5 neurons per hidden layer output layer has 2 neurons
        initial_weights = [[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],[[0,0,0,0,0,0],[0,0,0,0,0,0]]]
        initial_bias = [1,1,1,1]
        learning_rate = 0.9
        ann = maze_neural_network(layers,map,initial_weights,initial_bias,learning_rate)

        # train neural network
        #ann.randomize_weights()
        training_data = ann.get_training_data_from_file("training_data_set.txt")
        epochs = 10 # Complete pass of entire dataset
        ann.train_network(training_data,epochs)
        ann.save_weights()
        #ann.load_weights("neural_network_weights.txt") Load weights need fixing

        # set up visualizer
        visualize = illustrator()
        visualize.create_image(660,660)
        visualize.initialize_board(np.copy(self.map))

        # setup stats list
        episode_steps = []
        rando_scores = []
        ann_scores = []
        ann_reward_sum = 0
        winners = []

        # Run through experiment n times
        for episode in range(self.episodes):
            index = 0
            
            # get start state and choose start action for each mouse
            rando_state = self.initial_state
            rando_action = rando.choose_action(rando_state)

            # get initial sensor input for neural network
            ann_state = [self.initial_state[0],self.initial_state[1]]
            nose_direction, nose_distance = self.find_closest_cheese(ann_state)
            eye_direction, eye_distance = self.find_visible_cheese(ann_state)
            # get element in all four directions
            row = ann_state[1]
            col = ann_state[0]
            if row-1 >= 0:
                up = self.map[col][row-1]
                up = self.determine_direction_value(up)
            else:
                up = -1
            if row +1 <= len(self.map):
                down = self.map[col][row+1]
                down = self.determine_direction_value(down)
            else:
                down = -1
            if col -1 >= 0:
                left = self.map[col-1][row]
                left = self.determine_direction_value(left)
            else:
                left = -1
            if col +1 <= len(self.map):
                right = self.map[col+1][row]
                right = self.determine_direction_value(right)
            else:
                right = -1
            ann_action_num = ann.forward_pass([eye_direction,eye_distance,nose_direction,nose_distance,up,down,left,right])
            ann_action = ann.convert_output_to_maze_action(ann_action_num)
            cheese_count = self.remaining_cheese()
            rando_reward = 0
            ann_reward = 0
            step_count = 0
            # show map
            print("Episode: %i" %episode)
            if ann_state == rando_state:
                visualize.update_board(ann_state[0],ann_state[1],(255,150,255))
            else:
                visualize.update_board(rando_state[0],rando_state[1],(255,0,0))
                visualize.add_mouse(ann_state[0],ann_state[1],(0,150,255))
            visualize.show_board()
        
            # make steps until all cheese has been taken
            while cheese_count > 0:
                # Check if within wall
                if self.map[rando_state[1]][rando_state[0]] == "#":
                    print("ERROR: Rando stuck in wall")
                    visualize.show_board()
                    cv2.waitKey(0)
                if self.map[ann_state[0]][ann_state[1]] == "#":
                    print("ERROR: ANN stuck in wall")
                    visualize.show_board()
                    cv2.waitKey(0)

                # take action
                new_rando_state,rando_reward = rando.take_action(rando_action,rando_state)
                # now ANN
                new_ann_state,ann_reward = self.check_action(ann_action,ann_state,ann_action_num)
                step_count = step_count+1
                # show map
                if  new_ann_state == new_rando_state:
                    visualize.update_board(new_ann_state[1],new_ann_state[0],(255,150,255))
                else:
                    visualize.update_board(new_rando_state[0],new_rando_state[1],(255,0,0))
                    visualize.add_mouse(new_ann_state[1],new_ann_state[0],(0,0,100))
                visualize.show_board()

                # Reward handling
                if  ann_reward > 0 or rando_reward > 0:
                    if  new_ann_state == new_rando_state:
                        rando_reward = rando_reward/2
                        ann_reward = ann_reward/2
                        rando.give_reward(rando_reward)
                        ann_reward_sum += ann_reward
                        visualize.remove_goal(new_ann_state[1],new_ann_state[0])
                    else:
                        if  ann_reward > 0:
                            rando.remove_reward([new_ann_state[1],new_ann_state[0]])
                            visualize.remove_goal(new_ann_state[1],new_ann_state[0])
                        if rando_reward > 0:
                            rando.give_reward(rando_reward)
                            visualize.remove_goal(new_rando_state[0],new_rando_state[1])

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                # choose next action
                new_rando_action = rando.choose_action(new_rando_state)
                nose_direction, nose_distance = self.find_closest_cheese([new_ann_state[0],new_ann_state[1]])
                eye_direction, eye_distance = self.find_visible_cheese([new_ann_state[0],new_ann_state[1]])
                # get element in all four directions
                row = new_ann_state[0]
                col = new_ann_state[1]
                if row-1 >= 0:
                    up = self.map[row-1][col]
                    up = self.determine_direction_value(up)
                else:
                    up = -1
                if row +1 <= len(self.map[0]):
                    print(self.map)
                    down = self.map[row+1][col]
                    down = self.determine_direction_value(down)
                else:
                    down = -1
                if col -1 >= 0:
                    left = self.map[row][col-1]
                    left = self.determine_direction_value(left)
                else:
                    left = -1
                if col +1 <= len(self.map):
                    right = self.map[row][col+1]
                    right = self.determine_direction_value(right)
                else:
                    right = -1
                new_ann_action = ann.forward_pass([eye_direction,eye_distance,nose_direction,nose_distance,up,down,left,right])
                ann_action_num = new_ann_action
                new_ann_action = ann.convert_output_to_maze_action(new_ann_action)

                # Clear old spot
                self.map[ann_state[0]][ann_state[1]] = " "
                self.map[rando_state[1]][rando_state[0]] = " "

                # update state, action, index and map
                rando_state = new_rando_state
                rando_action = new_rando_action
                ann_state = new_ann_state
                ann_action = new_ann_action
                index = index+1
                self.map[ann_state[0]][ann_state[1]] = "s"
                self.map[rando_state[1]][rando_state[0]] = "r"
                cheese_count = self.remaining_cheese()

            # Post test results
            rando_score = rando.get_score()
            print("Rando result: %f cheese" %rando_score)
            print("ANN result: %f cheese" %ann_reward_sum)
            print("Total steps taken %i" %step_count)
            self.map = np.copy(self.original_map)
            episode_steps.append(step_count)
            rando.reset_map()
            visualize.reset_board()
            rando.reset_score()
            ann_scores.append(ann_reward_sum)
            ann_reward_sum = 0
            rando_scores.append(rando_score)
            if rando_score == ann_reward_sum:
                print("Its a tie!")
                winners.append("tie")
            elif rando_score > ann_reward_sum:
                print("Random mouse wins!")
                winners.append("random")
            else:
                print("ANN mouse wins!")
                winners.append("ANN")
        
        return winners,rando_scores,ann_scores

    def expected_vs_sarsa(self,epsilon,learning_rate,gamma,expected_epsilon,expected_learning_rate,expected_gamma,Q_file = 0,Q_file_expected = 0,load = 1,):
        # initialize classes
        sarsa = maze_sarsa()
        expected_sarsa = maze_expected_sarsa()
        rando = random_mouse(np.copy(self.original_map),self.initial_state)

        # give classes needed parameters
        sarsa.set_map(np.copy(self.map))
        sarsa.initialize_parameters(epsilon=epsilon,learning_rate=learning_rate,gamma=gamma, state=self.initial_state,move_penalty=-1)
        expected_sarsa.set_map(np.copy(self.map))
        expected_sarsa.initialize_parameters(epsilon=expected_epsilon,learning_rate=expected_learning_rate,gamma=expected_gamma, state=self.initial_state,move_penalty=-1)
        if load == 1:
            sarsa.load_Q_from_txt(Q_file)
            expected_sarsa.load_Q_from_txt(Q_file_expected)

        # set up visualizer
        visualize = illustrator()
        visualize.create_image(660,660)
        visualize.initialize_board(np.copy(self.map))

        # setup stats list
        episode_steps = []
        rando_scores = []
        expected_sarsa_scores = []
        sarsa_scores = []
        winners = []

        video = cv2.VideoWriter("sarsa_versus_expected.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 15, (660, 660))   

        # stop expected sarsa backtracking with punishment
        punishment = -10
        two_states_ago = 0
        two_states_ago_expected = 0

        # Run through experiment n times
        for episode in range(self.episodes):
            index = 0
            
            # get start state and choose start action for each mouse
            sarsa_state = self.initial_state
            expected_sarsa_state = self.initial_state
            rando_state = self.initial_state
            sarsa_action = sarsa.select_action(sarsa_state)
            expected_sarsa_action = expected_sarsa.select_action(expected_sarsa_state)
            rando_action = rando.choose_action(rando_state)


            cheese_count = self.remaining_cheese()
            expected_sarsa_reward = 0
            sarsa_reward = 0
            rando_reward = 0
            step_count = 0
            # show map
            print("Episode: %i" %episode)
            if expected_sarsa_state == rando_state == sarsa_state:
                visualize.update_board(expected_sarsa_state[0],expected_sarsa_state[1],(255,100,250))
            elif expected_sarsa_state == rando_state:
                visualize.update_board(expected_sarsa_state[0],expected_sarsa_state[1],(255,0,100))
                visualize.add_mouse(sarsa_state[0],sarsa_state[1],(0,0,255))
            elif sarsa_state == rando_state:
                visualize.update_board(sarsa_state[0],sarsa_state[1],(255,0,255))
                visualize.add_mouse(expected_sarsa_state[0],expected_sarsa_state[1],(0,0,100))
            elif sarsa_state == expected_sarsa_state:
                visualize.update_board(sarsa_state[0],sarsa_state[1],(0,100,255))
                visualize.add_mouse(rando_state[0],rando_state[1],(255,0,0))
            else:
                visualize.update_board(rando_state[0],rando_state[1],(255,0,0))
                visualize.add_mouse(expected_sarsa_state[0],expected_sarsa_state[1],(0,0,100))
                visualize.add_mouse(sarsa_state[0],sarsa_state[1],(0,0,255))
            visualize.show_board()
            # add to video
            image = visualize.get_image()
            video.write(image)
        
            # make steps until all cheese has been taken
            while cheese_count > 0:
                # Check if within wall
                if self.map[expected_sarsa_state[1]][expected_sarsa_state[0]] == "#":
                    print("ERROR: Expected sarsa stuck in wall")
                    visualize.show_board()
                    cv2.waitKey(0)
                elif self.map[sarsa_state[1]][sarsa_state[0]] == "#":
                    print("ERROR: Sarsa stuck in wall")
                    visualize.show_board()
                    cv2.waitKey(0)
                elif self.map[rando_state[1]][rando_state[0]] == "#":
                    print("ERROR: Rando stuck in wall")
                    visualize.show_board()
                    cv2.waitKey(0)

                # take action
                new_sarsa_state,sarsa_reward = sarsa.perform_action(sarsa_action)
                new_expected_sarsa_state,expected_sarsa_reward = expected_sarsa.perform_action(expected_sarsa_action)
                new_rando_state,rando_reward = rando.take_action(rando_action,rando_state)
                step_count = step_count+1
                # show map
                if expected_sarsa_state == rando_state == sarsa_state:
                    visualize.update_board(expected_sarsa_state[0],expected_sarsa_state[1],(255,100,250))
                elif expected_sarsa_state == rando_state:
                    visualize.update_board(expected_sarsa_state[0],expected_sarsa_state[1],(255,0,100))
                    visualize.add_mouse(sarsa_state[0],sarsa_state[1],(0,0,255))
                elif sarsa_state == rando_state:
                    visualize.update_board(sarsa_state[0],sarsa_state[1],(255,0,255))
                    visualize.add_mouse(expected_sarsa_state[0],expected_sarsa_state[1],(0,0,100))
                elif sarsa_state == expected_sarsa_state:
                    visualize.update_board(sarsa_state[0],sarsa_state[1],(0,100,255))
                    visualize.add_mouse(rando_state[0],rando_state[1],(255,0,0))
                else:
                    visualize.update_board(rando_state[0],rando_state[1],(255,0,0))
                    visualize.add_mouse(expected_sarsa_state[0],expected_sarsa_state[1],(0,0,100))
                    visualize.add_mouse(sarsa_state[0],sarsa_state[1],(0,0,255))
                visualize.show_board()
                # add to video
                image = visualize.get_image()
                video.write(image)

                # Reward handling
                if new_expected_sarsa_state == two_states_ago_expected:
                    expected_sarsa_reward = punishment
                if new_sarsa_state == two_states_ago:
                    sarsa_reward = punishment

                if expected_sarsa_reward > 0 or rando_reward > 0 or sarsa_reward > 0:
                    if new_expected_sarsa_state == new_rando_state == new_sarsa_state:
                        sarsa_reward = sarsa_reward/3
                        expected_sarsa_reward = expected_sarsa_reward/3
                        rando_reward = rando_reward/3
                        rando.give_reward(rando_reward)
                        expected_sarsa.give_score(expected_sarsa_reward)
                        sarsa.give_score(sarsa_reward)
                        visualize.remove_goal(new_expected_sarsa_state[0],new_expected_sarsa_state[1])
                    elif new_expected_sarsa_state == new_rando_state:
                        expected_sarsa_reward = expected_sarsa_reward/2
                        rando_reward = rando_reward/2
                        rando.give_reward(rando_reward)
                        expected_sarsa.give_score(expected_sarsa_reward)
                        visualize.remove_goal(new_expected_sarsa_state[0],new_expected_sarsa_state[1])
                        if sarsa_reward > 0:
                            sarsa.give_score(sarsa_reward)
                            visualize.remove_goal(new_sarsa_state[0],new_sarsa_state[1])
                    elif new_sarsa_state == new_rando_state:
                        sarsa_reward = sarsa_reward/2
                        rando_reward = rando_reward/2
                        rando.give_reward(rando_reward)
                        visualize.remove_goal(new_sarsa_state[0],new_sarsa_state[1])
                        if expected_sarsa_reward > 0:
                            expected_sarsa.give_score(expected_sarsa_reward)
                            visualize.remove_goal(new_expected_sarsa_state[0],new_expected_sarsa_state[1])
                        sarsa.give_score(sarsa_reward)
                    elif new_sarsa_state == new_expected_sarsa_state:
                        sarsa_reward = sarsa_reward/2
                        expected_sarsa_reward = expected_sarsa_reward/2
                        visualize.remove_goal(new_sarsa_state[0],new_sarsa_state[1])
                        if rando_reward > 0:
                            rando.give_reward(rando_reward)
                            visualize.remove_goal(new_rando_state[0],new_rando_state[1])
                        expected_sarsa.give_score(expected_sarsa_reward)
                        sarsa.give_score(sarsa_reward)
                    else:
                        if expected_sarsa_reward > 0:
                            rando.remove_reward(new_expected_sarsa_state)
                            sarsa.remove_reward(new_expected_sarsa_state)
                            expected_sarsa.give_score(expected_sarsa_reward)
                            visualize.remove_goal(new_expected_sarsa_state[0],new_expected_sarsa_state[1])
                        if rando_reward > 0:
                            expected_sarsa.remove_reward(new_rando_state)
                            sarsa.remove_reward(new_rando_state)
                            rando.give_reward(rando_reward)
                            visualize.remove_goal(new_rando_state[0],new_rando_state[1])
                        if sarsa_reward > 0:
                            expected_sarsa.remove_reward(new_sarsa_state)
                            rando.remove_reward(new_sarsa_state)
                            sarsa.give_score(sarsa_reward)
                            visualize.remove_goal(new_sarsa_state[0],new_sarsa_state[1])
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                # choose next action
                new_sarsa_action = sarsa.select_action(new_sarsa_state)
                new_expected_sarsa_action = expected_sarsa.select_action(new_expected_sarsa_state)
                new_rando_action = rando.choose_action(new_rando_state)
                # Clear old spot
                self.map[sarsa_state[1]][sarsa_state[0]] = " "
                self.map[expected_sarsa_state[1]][expected_sarsa_state[0]] = " "
                self.map[rando_state[1]][rando_state[0]] = " "

                # get action probabilities sum
                actions_sum = expected_sarsa.action_sum(new_expected_sarsa_state)
                # Update Q
                expected_sarsa.update_Q(expected_sarsa_state,new_expected_sarsa_state,expected_sarsa_reward,expected_sarsa_action,new_expected_sarsa_action,actions_sum)
                sarsa.update_Q(sarsa_state,new_sarsa_state,sarsa_reward,sarsa_action,new_sarsa_action)

                # update state, action, index and map
                two_states_ago_expected = expected_sarsa_state
                expected_sarsa_state = new_expected_sarsa_state
                expected_sarsa_action = new_expected_sarsa_action
                two_states_ago = sarsa_state
                sarsa_state = new_sarsa_state
                sarsa_action = new_sarsa_action
                rando_state = new_rando_state
                rando_action = new_rando_action
                index = index+1
                self.map[expected_sarsa_state[1]][expected_sarsa_state[0]] = "es"
                self.map[sarsa_state[1]][sarsa_state[0]] = "s"
                self.map[rando_state[1]][rando_state[0]] = "r"
                cheese_count = self.remaining_cheese()

            # Post test results
            expected_sarsa_score = expected_sarsa.get_score()
            sarsa_score = sarsa.get_score()
            rando_score = rando.get_score()
            print("Sarsa result: %f cheese" %sarsa_score)
            print("Expected sarsa result: %f cheese" %expected_sarsa_score)
            print("Rando result: %f cheese" %rando_score)
            print("Total steps taken %i" %step_count)
            self.map = np.copy(self.original_map)
            episode_steps.append(step_count)
            expected_sarsa.reset_map()
            sarsa.reset_map()
            rando.reset_map()
            visualize.reset_board()
            sarsa.reset_score()
            expected_sarsa.reset_score()
            rando.reset_score()
            if load == 1:
                expected_sarsa.load_Q_from_txt(Q_file)
                sarsa.load_Q_from_txt(Q_file)
            rando_scores.append(rando_score)
            expected_sarsa_scores.append(expected_sarsa_score)
            sarsa_scores.append(sarsa_score)
            if rando_score == expected_sarsa_score == sarsa_score:
                print("Its a three way tie!")
                winners.append("tie")
            elif rando_score > expected_sarsa_score and rando_score > sarsa_score:
                print("Random mouse wins!")
                winners.append("random")
            elif sarsa_score > expected_sarsa_score and sarsa_score > rando_score:
                print("Sarsa mouse wins!")
                winners.append("sarsa")
            elif expected_sarsa_score > sarsa_score and expected_sarsa_score > rando_score:
                print("Expected sarsa mouse wins!")
                winners.append("expected sarsa")
            else:
                print("Two way tie")
                winners.append("tie")
            # end video
            # Deallocating memories taken for window creation 
            cv2.destroyAllWindows()  
            video.release()  # releasing the video generated 
        return winners,rando_scores,expected_sarsa_scores,sarsa_score,episode_steps

# Test create maze manualy:
mazes = maze_creator()
mazes.create_maze_size(11,11)
mazes.select_maze(0)
mazes.insert_cheese([4,4,8],[[1,1],[9,9],[7,7]])
mazes.insert_edge_walls()
mazes.insert_spawn([5,5])
walls = [[4,4],[6,6],[4,6],[6,4],[4,3],[6,7],[6,8],[7,8],[8,8],[9,8],[7,6],[8,6],[7,4],[9,4]]
mazes.insert_walls(walls)

# Test create maze from template
maze = [["#","#","#","#","#","#","#","#","#","#","#"],
        ["#","8"," "," "," "," "," "," "," ","2","#"],
        ["#","#","#","#","#"," ","#","#","#"," ","#"],
        ["#","4"," ","2","#"," "," ","4"," "," ","#"],
        ["#","#"," ","#","#"," ","#","#","#"," ","#"],
        ["#"," "," "," "," ","S"," "," "," "," ","#"],
        ["#"," ","#","#","#"," ","#","#"," ","#","#"],
        ["#"," "," ","4"," "," ","#","2"," ","4","#"],
        ["#"," ","#","#","#"," ","#","#","#","#","#"],
        ["#","2"," "," "," "," "," "," "," ","8","#"],
        ["#","#","#","#","#","#","#","#","#","#","#"]]

training_maze = [["#","#","#","#","#","#","#","#","#","#","#"],
        ["#","8"," "," "," "," "," "," "," ","2","#"],
        ["#","#","#","#","#"," ","#","#","#"," ","#"],
        ["#","4"," ","2","#"," "," ","4"," "," ","#"],
        ["#","#"," ","#","#"," ","#","#","#"," ","#"],
        ["#"," "," "," "," "," "," "," "," "," ","#"],
        ["#"," ","#","#","#"," ","#","#"," ","#","#"],
        ["#"," "," ","4"," "," ","#","2"," ","4","#"],
        ["#"," ","#","#","#"," ","#","#","#","#","#"],
        ["#","2"," "," "," "," "," "," "," ","8","#"],
        ["#","#","#","#","#","#","#","#","#","#","#"]]

ANN_training_maze = [["#","#","#","#","#","#","#","#","#","#","#"],
                    ["#","8"," "," "," ","#"," "," ","2"," ","#"],
                    ["#"," ","#","#"," ","#","8"," "," "," ","#"],
                    ["#"," ","#","4"," ","#","#","#"," "," ","#"],
                    ["#"," ","#"," "," ","4"," "," ","2"," ","#"],
                    ["#"," ","#"," "," "," ","#","#"," "," ","#"],
                    ["#","4","#"," "," "," ","#"," "," "," ","#"],
                    ["#"," "," "," ","4"," ","4"," ","#"," ","#"],
                    ["#"," ","8","#","#","#"," "," ","#","#","#"],
                    ["#"," "," ","#"," ","2"," "," ","2"," ","#"],
                    ["#","#","#","#","#","#","#","#","#","#","#"]]

mazes.insert_maze(maze)
mazes.insert_maze(training_maze)
mazes.insert_maze(ANN_training_maze)
mazes.select_maze(1)

# get map
map = mazes.get_maze(1)

# visualize text
mazes.visualize_maze(1)
mazes.visualize_maze(0)

# illustrate
visualize = illustrator()
visualize.create_image(660,660)
visualize.initialize_board(np.copy(map))
visualize.show_board()

# get map
map = mazes.get_maze(2)

# train sarsa 0.1 epsilon
#sarsa_trainer = maze_sarsa()
#test = experiment()
#test.set_parameters(map,1000) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.1,0.0,0.2,0.8)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_epsilon_0_1_table")

# train sarsa 0.05 epsilon
#Q_sarsa_Result,steps = test.train_sarsa(0.05,0.0,0.2,0.8)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_epsilon_0_0_5_table")

# train sarsa 0.2 epsilon
#Q_sarsa_Result,steps = test.train_sarsa(0.2,0.0,0.2,0.8)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_epsilon_0_2_table")

# train sarsa 0.8
#sarsa_trainer = maze_sarsa()
#test = experiment()
#test.set_parameters(map,1000) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.2,0.0,0.8,0.8)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_learning_0_8_table")

# train sarsa 0.5 learning rate
#Q_sarsa_Result,steps = test.train_sarsa(0.2,0.0,0.5,0.8)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_learning_0_5_table")

# train sarsa 0.2 learning rate
#Q_sarsa_Result,steps = test.train_sarsa(0.2,0.0,0.2,0.8)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_learning_0_2_table")

# train sarsa gamma 0.8
#sarsa_trainer = maze_sarsa()
#test = experiment()
#test.set_parameters(map,1000) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.2,0.0,0.5,0.8)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_gamma_0_8_table")

# train sarsa gamma 0.5
#Q_sarsa_Result,steps = test.train_sarsa(0.2,0.0,0.5,0.5)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_gamma_0_5_table")

# train sarsa gamma 0.2
#Q_sarsa_Result,steps = test.train_sarsa(0.2,0.0,0.5,0.2)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_gamma_0_2_table")

# train sarsa 100 episodes
#sarsa_trainer = maze_sarsa()
#test = experiment()
#test.set_parameters(map,100) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.05,0.0,0.5,0.2)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_100_episodes_table")

# train sarsa gamma 500 episodes
#test.set_parameters(map,500) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.05,0.0,0.5,0.2)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_500_episodes_table")

#test.set_parameters(map,1000) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.05,0.0,0.5,0.2)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_1000_episodes_table")

#test.set_parameters(map,1500) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.05,0.0,0.5,0.2)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_1500_episodes_table")

#test.set_parameters(map,2000) # map, episodes
#Q_sarsa_Result,steps = test.train_sarsa(0.05,0.0,0.5,0.2)
#sarsa_trainer.save_Q_table(Q_sarsa_Result, "sarsa_Q_2000_episodes_table")


# train expected sarsa 0.05 epsilon
#expected_sarsa_trainer = maze_expected_sarsa()
#test = experiment()
#test.set_parameters(map,1000) # map, episodes
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.2,0.8)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.05_epsilon")
# train expected sarsa 0.1 epsilon
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.1,0.0,0.2,0.8)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.1_epsilon")
# train expected sarsa 0.2 epsilon
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.2,0.0,0.2,0.8)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.2_epsilon")

# train expected sarsa 0.2 alpha
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.2,0.8)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.2_alpha")
# train expected sarsa 0.5 alpha
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.5,0.8)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.5_alpha")
# train expected sarsa 0.8 alpha
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.8)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.8_alpha")

# train expected sarsa 0.2 gamma
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.2)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.2_gamma")
# train expected sarsa 0.5 gamma
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.5)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.5_gamma")
# train expected sarsa 0.8 gamma
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.8)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_0.8_gamma")

# train expected sarsa 100 episodes
#test.set_parameters(map,100) # map, episodes
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.5)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_100_episodes")
# train expected sarsa 500 episodes
#test.set_parameters(map,500) # map, episodes
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.5)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_500_episodes")
# test 1000 episodes
#test.set_parameters(map,1000) # map, episodes
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.5)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_1000_episodes")
# test 1500 episodes
#test.set_parameters(map,1500) # map, episodes
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.5)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_1500_episodes")
# test 1500 episodes
#test.set_parameters(map,2000) # map, episodes
#Q_expected_sarsa_Result = test.train_expected_sarsa(0.05,0.0,0.8,0.5)
#expected_sarsa_trainer.save_Q_table(Q_expected_sarsa_Result, "expected_sarsa_Q_table_2000_episodes")



# make training data set for neural network
#ANN_training_map = mazes.get_maze(3)
#test = experiment()
#test.set_parameters(ANN_training_map,1000)
#test.get_training_data_neural_network("training_data_set")

# Test sarsa versus random epsilon 0.2 won learning 0.5 won gamma 0.2 won 1500 episodes won
#map = mazes.get_maze(1)
#sarsa_test = experiment()
#sarsa_test.set_parameters(map,5000) # map, episode
#winners,rando_scores,sarsa_scores,episdoe_steps = sarsa_test.sarsa_vs_random(0.2,0.5,0.2,"sarsa_Q_1500_episodes_table.txt")

#total_sarsa_wins = 0
#total_random_wins = 0
#total_ties = 0
#total_sarsa_score = 0
#otal_random_score = 0
#open("Moving_wins.txt","w").close() # Resets file
#result_file = open("Moving_wins.txt","a")
#sarsa_moving_win = 0
#random_moving_win = 0
#moving_tie = 0
#for element in winners:
#    if element == "sarsa":
#        sarsa_moving_win +=1
#    elif element == "random":
#        random_moving_win += 1
#    elif element == "tie":
#        moving_tie += 1
#    result_file.write(str(sarsa_moving_win)+","+str(random_moving_win)+","+str(moving_tie)+"\n")
#result_file.write("\n")
#result_file.close()

# Test expected sarsa versus random epsilon 0.05 won alpha 0.8 won gamma 0.5 won 2000 episodes won
#map = mazes.get_maze(1)
#expected_sarsa_test = experiment()
#expected_sarsa_test.set_parameters(map,5000) # map, episode
#winners,rando_scores,expected_sarsa_scores,episode_steps = expected_sarsa_test.expected_sarsa_vs_random(0.05,0.8,0.5,"expected_sarsa_Q_table_2000_episodes.txt")

#total_expected_sarsa_wins = 0
#total_random_wins = 0
#total_ties = 0
#total_expected_sarsa_score = 0
#total_random_score = 0
#open("Moving_wins_expected.txt","w").close() # Resets file
#result_file = open("Moving_wins_expected.txt","a")
#expected_sarsa_moving_win = 0
#random_moving_win = 0
#moving_tie = 0
#for element in winners:
#    if element == "expected sarsa":
#        expected_sarsa_moving_win +=1
#    elif element == "random":
#        random_moving_win += 1
#    elif element == "tie":
#        moving_tie += 1
#    result_file.write(str(expected_sarsa_moving_win)+","+str(random_moving_win)+","+str(moving_tie)+"\n")
#result_file.write("\n")
#result_file.close()

#total_expected_sarsa_wins = 0
#total_random_wins = 0
#total_ties = 0
#total_expected_sarsa_score = 0
#total_random_score = 0
#open("Expected sarsa_versus_random_2000_episodes.txt","w").close() # Resets file
#result_file = open("Expected sarsa_versus_random_2000_episodes.txt","a")
#for element in winners:
#    if element == "expected sarsa":
#        total_expected_sarsa_wins +=1
#    elif element == "random":
#       total_random_wins += 1
#    elif element == "tie":
#        total_ties += 1
#for element in rando_scores:
#    result_file.write(str(element)+",")
#    total_random_score += int(element)
#result_file.write("\n")  
#for element in expected_sarsa_scores:
#    result_file.write(str(element)+",")
#    total_expected_sarsa_score += int(element)
#result_file.write("\n")
#for element in episode_steps:
#    result_file.write(str(element)+",")
#result_file.write("\n")
#result_file.write(str(total_expected_sarsa_wins))
#result_file.write("\n")
#result_file.write(str(total_expected_sarsa_score))
#result_file.write("\n")
#result_file.write(str(total_random_wins))
#result_file.write("\n")
#result_file.write(str(total_random_score))
#result_file.write("\n")
#result_file.write(str(total_ties))
#result_file.write("\n")
#result_file.close()


# Sarsa versus expected versus random
#map = mazes.get_maze(1)
#compare = experiment()
#compare.set_parameters(map,1) # map, episode
#winners,rando_scores,expected_sarsa_scores,sarsa_scores,episode_steps = compare.expected_vs_sarsa(0.2,0.5,0.2,0.05,0.8,0.5,"sarsa_Q_1500_episodes_table.txt","expected_sarsa_Q_table_2000_episodes.txt")

#open("Moving_wins_comparison.txt","w").close() # Resets file
#result_file = open("Moving_wins_comparison.txt","a")
#expected_sarsa_moving_win = 0
#sarsa_moving_win = 0
#random_moving_win = 0
#moving_tie = 0
#for element in winners:
#    if element == "expected sarsa":
#        expected_sarsa_moving_win +=1
#    elif element == "random":
#        random_moving_win += 1
#    elif element == "sarsa":
#        sarsa_moving_win += 1
#    elif element == "tie":
#        moving_tie += 1
#    result_file.write(str(expected_sarsa_moving_win)+","+str(sarsa_moving_win)+","+str(random_moving_win)+","+str(moving_tie)+"\n")
#result_file.write("\n")
#result_file.close()







# Test ANN versus random
map = mazes.get_maze(1)
ann_test = experiment()
ann_test.set_parameters(map,1) # map, episode
winners,rando_scores,ann_scores = ann_test.ANN_vs_random()

#total_ann_wins = 0
#total_random_wins = 0
#total_ties = 0
#total_ann_score = 0
#total_random_score = 0
#open("ANN_versus_random.txt","w").close() # Resets file
#result_file = open("ANN_versus_random.txt","a")
#result_file.write("winners:")
#for element in winners:
#    result_file.write(str(element)+",")
#    if element == "ANN":
#        total_ann_wins += 1
#    elif element == "random":
#        total_random_wins += 1
#    elif element == "tie":
#        total_ties += 1
#result_file.write("\nrandom score:")
#for element in rando_scores:
#    result_file.write(str(element)+",")
#    total_random_score += int(element)
#result_file.write("\nANN score:")  
#for element in ann_scores:
#    result_file.write(str(element)+",")
#    total_ann_score += int(element)
#result_file.write("\nANN total wins: ")
#result_file.write(str(total_ann_wins))
#result_file.write("\nANN total score: ")
#result_file.write(str(total_ann_score))
#result_file.write("\nrandom total wins: ")
#result_file.write(str(total_random_wins))
#result_file.write("\nrandom total score: ")
#result_file.write(str(total_random_score))
#result_file.write("\ntotal ties: ")
#result_file.write(str(total_ties))
#result_file.write("\n")
#result_file.close()




# input: Nearest cheese direction (0 = up, 1 = down, 2 = left, 3 = right), cheese distance (n: number of tiles), object up,down,left,right (-1 = wall, 0 = free, 1-9 = cheese)
#test_input = [2,1,-1,-1,2,0] # cheese to left
#desired_result = [1,0] # 0,0 = up 0,1 = down, 1,0 = left, 1,1 = right
#iterations = 1000
#neural_network.test_network(desired_result,test_input,iterations)




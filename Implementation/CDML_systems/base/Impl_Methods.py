import socket
import sys 
import time

import torch
from . import Base_Constants
import pickle


class Communication():
    def __init__(self, server_address=None):
        pass

    def receive_information(self):
        """
        timer is the time the system waits for new information before the server is closed. It is meassured in seconds.
        amount is the amount of information that should be received. It is measured in amount of connections
        If no values are specified exactly one connection will be used
        Else the system will check the timer and amount. As soon as one is reached the function returns all the connections
        This can however fail, since time isn't checked at all times.
        """

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1) # timeout for listening
        # Bind the socket to the port
        print (sys.stderr, 'starting up on %s port %s' % self.ip)
        sock.bind(self.ip)
        # Listen for incoming connections
        sock.listen(1)
        while True:
            # Wait for a connection
            try:
                connection, client_address = sock.accept()
                try:
                    print (sys.stderr, 'connection from', client_address)
                    complete_data = b''

                    # Receive the data in small chunks and retransmit it
                    while True:
                        data = connection.recv(4096)
                        if data:
                            connection.sendall(data)
                            complete_data += data
                        else:

                            print (sys.stderr, 'no more data from', client_address)
                            complete_data = pickle.loads(complete_data)
                            print("Received: " + complete_data.tag)
                            sock.close()
                            connection.close()
                            return complete_data
                except:
                    connection.close()
                    sock.close()
                    time.sleep(2)
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(self.ip)
                    # Listen for incoming connections
                    sock.listen(1)
                    print("Trying again!")
                finally:
                    # Clean up the connection
                    connection.close()
                    sock.close()
                    #return all_data_entries
            except socket.timeout:
                print("Socket timed out, no incoming connections.")
                time.sleep(2)
                    


    def send_information(self, message, communication_address, timer=None):
        """
        timer is here the time used to wait for a connection to another server
        """
        if timer is not None:
            start_time = time.time()
        else:
            start_time = time.time()
            timer = 1200.0
        
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        print (sys.stderr, 'connecting to %s port %s' % communication_address) 
        done = True
        while time.time()-start_time < timer:
            try:           
                sock.connect(communication_address)
                # Send data
                print (sys.stderr, 'sending "%s"' % message.tag)
                message_dumped = pickle.dumps(message)
                sock.sendall(message_dumped)

                # Look for the response
                amount_received = 0
                amount_expected = len(message_dumped)
                
                while amount_received < amount_expected:
                    data = sock.recv(4096)
                    amount_received += len(data)
                done = True 

            except:
                sock.close()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                time.sleep(2)
                print("Trying again!")
                done = False
            if done:
                print (sys.stderr, 'Closing socket')
                sock.close()
                break
        if not done:
            sock.close()
            print("No connection in time possible!")
            return None
       

class Repository():
    def __init__(self, CDMLSystem, CDMLSystemAlgorithm, interimResult, agentRequirements, communicationProtocoll):
        """
        CDMLSystem means what specific system of CDML will be used.
        CDMLSystemAlgorithm specifies which exact version of the CDML system will be done.
        interimResult means what update the agents will have to send
        agentRequirements refers to what an agent needs to fulfill to be accepted by the configurator, the agentRequirements should be in form of the class AgentRequirements
        communicationProtocoll refers to how the agent sends messages to the agents
        """
        self.CDMLSystem = CDMLSystem
        self.CDMLSystemAlgorithm = CDMLSystemAlgorithm
        self.interimResult = interimResult
        self.agentRequirements = agentRequirements
        self.communicationProtocoll = communicationProtocoll

class Application():
    def __init__(self, roles, id, ip, agentAttributes):
        """
        roles means the roles an agent wants to take
        id is a way to identify the agent. Should the id be already taken the application is refused
        ip the ip the agent has, so it can be reached with further information
        agentAttributes is the response to the agentRequirements from the Repository, so the cofigurator can decide on the application
        """
        self.roles = roles
        self.id = id
        self.ip = ip
        self.agentAttributes = agentAttributes

class Message():
    def __init__(self, message):
        """
        the message is required to be in the form of communicationProtocoll from Repository
        """
        self.message = message

    def toString(self):
        return str(self.message)
    
    
class CoalitionEntry():
    def __init__(self, id, ip, agent):
        self.id = id
        self.ip = ip
        self.agent = agent
    

class CoalitionEntryResponse():
    def __init__(self,result, id, ip, agent):
        self.result = result
        self.id = id
        self.ip = ip
        self.agent = agent

class Message():
    def __init__(self, tag, object):
        """
        The tag defines what kind of message is sent.\n
        The object is the message that is to be sent and received.
        """
        self.tag = tag
        self.object = object


class AgentRequirements():
    """
    This class is here to be a standard way to return a requirements for agents
    """
    def __init__(self, requirementsDefined=False):
        """
        requirementsDefined gives agents the opportunity to check wether the configurator has special requests. Default its set to False
        """
        self.requirementsDefined = requirementsDefined

class PurposeOfSystem():
    """
    This class is here to be a standard way to return a purpose of the system for agents
    """
    def __init__(self, requirementsDefined=False):
        """
        requirementsDefined gives agents the opportunity to check wether the configurator has special requests. Default its set to False
        """
        self.requirementsDefined = requirementsDefined

class Coalition():
    def __init__(self, purposeOfSystem, agentRequirements, repository):
        self.purposeOfSystem = purposeOfSystem
        self.agentRequirements = agentRequirements
        self.repository = repository

class InterimResult():
    """
    This class is a way to standardize the Interim Results
    """
    def __init__(self, requirementsDefined=False, results=None):
        """
        requirementsDefined gives agents the opportunity to check wether the configurator has special requests. Default its set to False
        results should be in Form of Base_Constants.interim_results where the first entry is a either model, split, or statistics from base_constant.possible_result
        """
        self.requirementsDefined=requirementsDefined
        if requirementsDefined:
            self.results = results

class AgentRoleConversion():
    def __init__(self, agent_name):
        self.roles = []
        for entry in Base_Constants.agent_tag_to_role: #TODO, validate whether a dict can be iterated like this
            if entry in agent_name:
                self.roles.append(Base_Constants.agent_tag_to_role[entry])

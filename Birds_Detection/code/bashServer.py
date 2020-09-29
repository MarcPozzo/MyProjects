#!/usr/bin/python
from http.server import SimpleHTTPRequestHandler,BaseHTTPRequestHandler,HTTPServer
import subprocess
import json
import os
import sys

# parameters
os.chdir('code/')
logFile = "../log/bashServer.log" # ajouter ici un timestamp
PORT_NUMBER = 8082

# clean-up signal files, just in case (reboot when running for example)
signalFolder = "signalFiles"
listFiles = os.listdir(signalFolder)
for signalFile in listFiles:
    if(signalFile != "README"):
        os.remove(signalFolder+"/"+signalFile)

# This class will handles any incoming request from
# the browser 
self = "lancerBoucle"
class myHandler(BaseHTTPRequestHandler):
        def do_POST(self):
                content_len = int(self.headers.get('content-length'))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode("utf-8"))
                print(data)

                # Use the post data
                signal = data["cmd"]
                shouldWait = True;
                stdoutBash = subprocess.PIPE
                if(signal == "startMotion"):
                    cmd = "motion ./motion.conf"
                elif(signal == "stopMotion"):
                    cmd = "pkill motion"
                elif(signal == "savePicture"):
                    cmd = "../bin/stillImage.sh"
                elif(signal == "energySavings"):
                    cmd = "../bin/economyMode.sh"
                elif(signal == "noEnergySavings"):
                    cmd = "../bin/normalMode.sh"
                elif(signal == "stopBoucle"):  
                    cmd = "" 
                    sF = open("signalFiles/signalfile.txt",'w')
                    sF.write("global stop\nstop=True")
                    sF.close()
                elif(signal == "Reboot"):
                    cmd = "sudo reboot"
                elif(signal == "miseAjour"):
                    cmd = "git pull"
                elif(signal == "MettreAlheure"):
                    cmd = "sudo hwclock -s"        
                elif(signal == "Shutdown"):
                    cmd = "sudo shutdown -h now"
                elif(signal == "lancerBoucleObs"):
                    cmd = "../bin/Boucle.py -t None &>>"+logFile
                    shouldWait = False
                    stdoutBash = f1                 
                elif(signal == "lancerBoucleAnal"):
                    cmd = "../bin/Boucle.py -t ssim &>>"+logFile
                    shouldWait = False
                    stdoutBash = f1                 
                else:
                    cmd =""

                p = subprocess.Popen(cmd, stdout=stdoutBash, stderr=stdoutBash, shell=True)
                print("p launched \n")
                if(shouldWait):
                    p_status = p.wait()
                    print("wait done \n")
                    (output, err) = p.communicate()
                else:
                    p_status = None
                    output = None
                    err = None
                
                print("Command output : ", output)
                print("Command exit status/return code : ", p_status)

                sys.stdout.flush()
                
                self.send_response(200)
                self.end_headers()
                
                # out = self.wfile.write(signal.encode("utf-8") + "\n")
                return(signal)

        def send_response(self, *args, **kwargs):
                SimpleHTTPRequestHandler.send_response(self, *args, **kwargs)
                self.send_header('Access-Control-Allow-Origin', '*')
                return("done")
try:
        # Create a web server and define the handler to manage the
        # incoming request
        print("Python bashServer.py starting")
        sys.stdout.flush()
        
        f1=open(logFile, 'w+')
        server = HTTPServer(('', PORT_NUMBER), myHandler)
        print('Started httpserver on port ' , PORT_NUMBER)

        # Wait forever for incoming http requests
        server.serve_forever()
        f1.close()

except KeyboardInterrupt:
        print('^C received, shutting down the web server')
        server.socket.close()

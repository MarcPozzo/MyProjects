<!DOCTYPE html>
<html>
    <head lang="en">
        <meta charset='utf-8'>
        <meta name="viewport" content="width=device-width,height=device-height, user-scalable=no" />
        <title>Choice of angle</title>

        <script src="base.js"></script>
        <link rel="stylesheet" href="base.css">

    </head>
    <body>
        C3PO: choose aerial view <a href="index.php">back</a>

        <div id='liveStream'>
<?php 
$mapFiles = scandir("maps/");
$outListFile = "";
$posURL = "positioning.php";
$initParams = $_SERVER['QUERY_STRING'];
for($iImage=0; $iImage < sizeof($mapFiles);$iImage++){
    $addIm = $mapFiles[$iImage];
    if(! in_array($addIm,array(".",".."))){
        $imageCode = '<img src="maps/'.$addIm.'" alt="'.$addIm.'" height="300" onclick/>';
        $linkImage = '<a href=\''.$posURL.'?'.$initParams.'&mapImage=maps/'.$addIm.'\' >'.$imageCode.'</a>';
        $divImage = '<div id="image'.$iImage.'">'.$linkImage.'</div>';

        $outListFile = $outListFile.$divImage;
    }
}
echo $outListFile;
?>

        <button type="submit" name="submit_show" id="submit_show" value="show" onclick="actionSend()">Done
        </button>
        <span id="result"></span>

        <script>
// set up the sending of commands to python
function sendToPython(value,response=respond) {
    if (window.XMLHttpRequest) {// code for IE7+, Firefox, Chrome, Opera, Safari
        xmlhttp = new XMLHttpRequest();
    }
    else {// code for IE6, IE5
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }
    // get time stamp
    var values = {"cmd":value};
    var myJsonString = JSON.stringify(values);
    xmlhttp.onreadystatechange = response;
    xmlhttp.open("POST", location.protocol+"//"+window.location.hostname+":8082", true);
    xmlhttp.send(myJsonString);
}
// receiving response after sending
function respond() {
    if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
        document.getElementById('result').innerHTML = xmlhttp.responseText;
    }
}

// send and next step
function actionSend(){
    var toSend = {positionG:positionG};
    var out = sendToPython("stopMotion");
    var out = sendToPython("savePicture",function(){
        var options = 'lat='+positionG.coords.latitude+'&long='+positionG.coords.longitude;
        window.location.replace("positionning.php?"+options);
    });
}

        </script>
    </body>
</html>


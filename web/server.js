const express = require("express");
const http = require("http");
const socket = require("socket.io");

const port = process.env.PORT || parseInt(process.argv[2]);

app = express();
app.use(express.static(__dirname + "/public"));

const server = http.createServer(app);
const io = socket(server);

io.on("connect", function (socket) {

});

let counter = 0;

app.get("/", function (req, res) {
    res.render("index.html");
})

server.listen(port);


//Multicast Client receiving sent messages
let PORT = 8123;
let MCAST_ADDR = "225.0.0.250"; //same mcast address as Server
let dgram = require('dgram');
let client = dgram.createSocket({type: 'udp4', reuseAddr: true})

client.on('listening', function () {
    let address = client.address();
    console.log('UDP Client listening on ' + address.address + ":" + address.port);
    client.setBroadcast(true)
    client.setMulticastTTL(128);
    client.addMembership(MCAST_ADDR);
});

let count = 0;
client.on('message', function (message, remote) {
    count++;
    //if (getRandomArbitrary(0, 100) >= 30) {
    let msg = message.toString();
    console.log(count);
    io.emit('multicast', msg);
    //}

});

function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

client.bind(PORT, "225.0.0.250");


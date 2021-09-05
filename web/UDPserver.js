const fs = require('fs');

//Multicast Server sending messages
let PORT = 8123;
let MCAST_ADDR = "225.0.0.250"; //not your IP and should be a Class D address, see http://www.iana.org/assignments/multicast-addresses/multicast-addresses.xhtml
let dgram = require('dgram');
let server = dgram.createSocket("udp4");
let rawdata = fs.readFileSync('animation.json');
let animation = JSON.parse(rawdata); // Array<THREE.AnimationClip>
const interval = 100; // interval of each update
const msgLimit = 50;

let currentIndex = new Array(animation.length).fill(0);
for (let i = 0; i < animation.length; i++) {
    animation[i]['size'] = Object.keys(animation[i].times).length;
    animation[i]['dimension'] = Object.keys(animation[i].values).length / animation[i].size; // preprocessing
}

server.bind(PORT, function () {
    server.setBroadcast(true);
    server.setMulticastTTL(128);
    server.addMembership(MCAST_ADDR);
});

let intervalCount = 0;
let componentCount = 0;
let count = 0;

setInterval(broadcastNew, interval);

function broadcastNew() {
    intervalCount++;
    let message = [];
    message.push({time: intervalCount * interval});
    let curTime = (intervalCount - 1) * interval / 1000;
    for (let i = 0; i < animation.length; i++) {
        let cur = {name: animation[i].name};
        while (animation[i].times[currentIndex[i]] <= curTime) {
            let vec = [];
            for (let j = 0; j < animation[i].dimension; j++) {
                vec.push(animation[i].values[currentIndex[i] * animation[i].dimension + j]);
            }
            cur[animation[i].times[currentIndex[i]]] = vec;
            currentIndex[i]++;
        }
        if (Object.keys(cur).length > 1) {
            if (componentCount + Object.keys(cur).length > msgLimit) {
                let msg = Buffer.from(JSON.stringify(message));
                //if (getRandomArbitrary(0, 100) != 0) server.send(msg, 0, msg.length, PORT, MCAST_ADDR); //for packet loss purpose
                server.send(msg, 0, msg.length, PORT, MCAST_ADDR);
                count++;
                console.log(count);
                message = [];
                message.push({time: (intervalCount - 1) * interval});
                componentCount = 0;
            }
            message.push(cur);
            componentCount += Object.keys(cur).length
        }
    }
    if (message.length) {
        let msg = Buffer.from(JSON.stringify(message));
        //if (getRandomArbitrary(0, 100) != 0) server.send(msg, 0, msg.length, PORT, MCAST_ADDR); //for packet loss purpose
        server.send(msg, 0, msg.length, PORT, MCAST_ADDR);
        count++;
        console.log(count);
    }


    if (curTime > 25) {
        intervalCount = 0;
        for (let i = 0; i < animation.length; i++) {
            currentIndex[i] = 0;
        }
    }
}

function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

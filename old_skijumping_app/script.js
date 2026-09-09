let canvas;
let context;
let trajectoryPoints = [];
let hillPoints = [];
let animationFrame = null;
let animationStart = null;
let messageHandlerRegistered = false;

const hillImage = new Image();
hillImage.src = "planica-three-quarter-v2.png";

const skierImage = new Image();
skierImage.src = "ski-jumper.png";

// Calibrated against planica-three-quarter-v2.png. Values are normalized
// canvas fractions per metre. Takeoff (0, 0, 0) is fixed to the lip, while
// (177, 0, -106) lies on the photographed landing slope.
const projection = {
    originX: 0.265,
    originY: 0.305,
    xToX: 0.00300,
    xToY: 0.0,
    yToX: -0.00420,
    yToY: 0.00340,
    zToX: 0.0,
    zToY: -0.00455,
};

function initializeCanvas() {
    canvas = document.getElementById("jump-canvas");
    if (!canvas) {
        setTimeout(initializeCanvas, 100);
        return;
    }
    context = canvas.getContext("2d");
    resizeCanvas();
    new ResizeObserver(resizeCanvas).observe(canvas.parentElement);
    hillImage.addEventListener("load", redrawStaticScene);
    skierImage.addEventListener("load", redrawStaticScene);
}

function resizeCanvas() {
    if (!canvas || !context) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * pixelRatio);
    canvas.height = Math.round(bounds.height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    redrawStaticScene();
}

function canvasSize() {
    const bounds = canvas.getBoundingClientRect();
    return { width: bounds.width, height: bounds.height };
}

function projectPoint(x, y, z) {
    const { width, height } = canvasSize();
    return {
        x: width * (projection.originX + Number(x) * projection.xToX
            + Number(y) * projection.yToX + Number(z) * projection.zToX),
        y: height * (projection.originY + Number(x) * projection.xToY
            + Number(y) * projection.yToY + Number(z) * projection.zToY),
    };
}

function drawBackground() {
    const { width, height } = canvasSize();
    context.clearRect(0, 0, width, height);
    if (hillImage.complete && hillImage.naturalWidth > 0) {
        context.drawImage(hillImage, 0, 0, width, height);
        context.fillStyle = "rgba(235, 244, 250, 0.10)";
        context.fillRect(0, 0, width, height);
        return;
    }
    const gradient = context.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, "#b9d6ea");
    gradient.addColorStop(1, "#f1f6f9");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);
}

function drawArrow(from, to, colour, label) {
    const angle = Math.atan2(to.y - from.y, to.x - from.x);
    const head = 7;
    context.save();
    context.strokeStyle = colour;
    context.fillStyle = colour;
    context.lineWidth = 2.2;
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.beginPath();
    context.moveTo(to.x, to.y);
    context.lineTo(to.x - head * Math.cos(angle - Math.PI / 6), to.y - head * Math.sin(angle - Math.PI / 6));
    context.lineTo(to.x - head * Math.cos(angle + Math.PI / 6), to.y - head * Math.sin(angle + Math.PI / 6));
    context.closePath();
    context.fill();
    context.font = "bold 12px sans-serif";
    context.shadowColor = "rgba(255, 255, 255, 0.95)";
    context.shadowBlur = 3;
    context.fillText(label, to.x + 7, to.y - 5);
    context.restore();
}

function drawCoordinateAxes() {
    const origin = projectPoint(0, 0, 0);
    context.save();
    context.fillStyle = "rgba(255, 255, 255, 0.86)";
    context.beginPath();
    context.arc(origin.x, origin.y, 5, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "#243746";
    context.lineWidth = 1.5;
    context.stroke();
    context.restore();
    drawArrow(origin, projectPoint(28, 0, 0), "#d74332", "X");
    drawArrow(origin, projectPoint(0, 12, 0), "#198754", "Y");
    drawArrow(origin, projectPoint(0, 0, 20), "#1368ce", "Z");
    context.save();
    context.font = "11px sans-serif";
    context.fillStyle = "#1e303d";
    context.textAlign = "center";
    context.shadowColor = "white";
    context.shadowBlur = 4;
    context.fillText("takeoff (0, 0, 0)", origin.x, origin.y - 14);
    context.restore();
}

function hillZAt(x) {
    if (hillPoints.length === 0) return 0;
    if (x <= Number(hillPoints[0][0])) return Number(hillPoints[0][1]);
    for (let index = 1; index < hillPoints.length; index++) {
        const left = hillPoints[index - 1];
        const right = hillPoints[index];
        if (x <= Number(right[0])) {
            const span = Number(right[0]) - Number(left[0]);
            const fraction = span === 0 ? 0 : (x - Number(left[0])) / span;
            return Number(left[1]) + fraction * (Number(right[1]) - Number(left[1]));
        }
    }
    return Number(hillPoints[hillPoints.length - 1][1]);
}

function drawProjectedHill() {
    if (hillPoints.length < 2) return;
    const halfWidth = 12;
    context.save();

    context.beginPath();
    hillPoints.forEach((point, index) => {
        const screen = projectPoint(point[0], -halfWidth, point[1]);
        if (index === 0) context.moveTo(screen.x, screen.y);
        else context.lineTo(screen.x, screen.y);
    });
    [...hillPoints].reverse().forEach((point) => {
        const screen = projectPoint(point[0], halfWidth, point[1]);
        context.lineTo(screen.x, screen.y);
    });
    context.closePath();
    context.fillStyle = "rgba(225, 242, 252, 0.16)";
    context.fill();

    for (const lateral of [-halfWidth, 0, halfWidth]) {
        context.beginPath();
        hillPoints.forEach((point, index) => {
            const screen = projectPoint(point[0], lateral, point[1]);
            if (index === 0) context.moveTo(screen.x, screen.y);
            else context.lineTo(screen.x, screen.y);
        });
        context.strokeStyle = lateral === 0 ? "rgba(35, 65, 82, 0.78)" : "rgba(55, 91, 112, 0.38)";
        context.lineWidth = lateral === 0 ? 2.4 : 1.2;
        context.stroke();
    }

    for (let x = 25; x <= 175; x += 25) {
        const z = hillZAt(x);
        const left = projectPoint(x, -halfWidth, z);
        const right = projectPoint(x, halfWidth, z);
        context.beginPath();
        context.moveTo(left.x, left.y);
        context.lineTo(right.x, right.y);
        context.strokeStyle = "rgba(55, 91, 112, 0.32)";
        context.lineWidth = 1;
        context.stroke();
        const centre = projectPoint(x, 0, z);
        context.font = "10px sans-serif";
        context.fillStyle = "rgba(28, 54, 70, 0.9)";
        context.textAlign = "center";
        context.fillText(`${x} m`, centre.x + 3, centre.y + 13);
    }
    context.restore();
}

function drawTrajectory(lastIndex = trajectoryPoints.length - 1) {
    if (trajectoryPoints.length < 2 || lastIndex < 1) return;
    const end = Math.min(lastIndex, trajectoryPoints.length - 1);
    const first = trajectoryPoints[0];
    const firstScreen = projectPoint(first[0], first[1], first[2]);
    context.save();
    context.beginPath();
    context.moveTo(firstScreen.x, firstScreen.y);
    for (let index = 1; index <= end; index++) {
        const point = trajectoryPoints[index];
        const screen = projectPoint(point[0], point[1], point[2]);
        context.lineTo(screen.x, screen.y);
    }
    context.strokeStyle = "#075de8";
    context.lineWidth = 3.2;
    context.shadowColor = "rgba(255, 255, 255, 0.95)";
    context.shadowBlur = 4;
    context.stroke();
    context.restore();
}

function drawSkier(index) {
    if (trajectoryPoints.length === 0) return;
    const safeIndex = Math.max(0, Math.min(index, trajectoryPoints.length - 1));
    const point = trajectoryPoints[safeIndex];
    const screen = projectPoint(point[0], point[1], point[2]);
    const neighbourIndex = safeIndex < trajectoryPoints.length - 1 ? safeIndex + 1 : Math.max(0, safeIndex - 1);
    const neighbour = trajectoryPoints[neighbourIndex];
    const neighbourScreen = projectPoint(neighbour[0], neighbour[1], neighbour[2]);
    let angle = Math.atan2(neighbourScreen.y - screen.y, neighbourScreen.x - screen.x);
    if (neighbourIndex < safeIndex) angle += Math.PI;
    context.save();
    context.translate(screen.x, screen.y);
    context.rotate(angle);
    if (skierImage.complete && skierImage.naturalWidth > 0) {
        const width = Math.max(52, Math.min(82, canvasSize().width * 0.078));
        const height = width * (skierImage.naturalHeight / skierImage.naturalWidth);
        context.drawImage(skierImage, -width * 0.46, -height * 0.58, width, height);
    } else {
        context.beginPath();
        context.arc(0, 0, 6, 0, Math.PI * 2);
        context.fillStyle = "#e52222";
        context.fill();
    }
    context.restore();
}

function drawScene(skierIndex, trajectoryIndex) {
    if (!canvas || !context) return;
    drawBackground();
    drawProjectedHill();
    drawCoordinateAxes();
    drawTrajectory(trajectoryIndex);
    drawSkier(skierIndex);
}

function redrawStaticScene() {
    if (!canvas || !context) return;
    drawScene(0, Math.max(0, trajectoryPoints.length - 1));
}

function startAnimation(coords) {
    if (Array.isArray(coords) && coords.length > 0) trajectoryPoints = coords;
    if (trajectoryPoints.length === 0) return;
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    animationStart = null;
    const millisecondsPerPoint = 45;
    function animate(timestamp) {
        if (animationStart === null) animationStart = timestamp;
        const index = Math.min(Math.floor((timestamp - animationStart) / millisecondsPerPoint), trajectoryPoints.length - 1);
        drawScene(index, index);
        if (index < trajectoryPoints.length - 1) animationFrame = requestAnimationFrame(animate);
        else animationFrame = null;
    }
    animationFrame = requestAnimationFrame(animate);
}

function registerShinyBridge() {
    if (messageHandlerRegistered) return;
    if (typeof Shiny === "undefined") {
        setTimeout(registerShinyBridge, 100);
        return;
    }
    Shiny.addCustomMessageHandler("render_trajectory", function(message) {
        if (!message || !Array.isArray(message.coords)) return;
        trajectoryPoints = message.coords;
        hillPoints = Array.isArray(message.hill) ? message.hill : [];
        redrawStaticScene();
    });
    Shiny.addCustomMessageHandler("start_trajectory", function(message) {
        startAnimation(message && message.coords);
    });
    messageHandlerRegistered = true;
}

registerShinyBridge();
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeCanvas, { once: true });
} else {
    initializeCanvas();
}

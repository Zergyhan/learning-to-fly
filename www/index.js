import * as sim from "lib-simulation-wasm";

CanvasRenderingContext2D.prototype.drawTriangle = function (x, y, size, rotation) {
    this.beginPath();

    this.moveTo(
        x - Math.sin(rotation) * size * 1.5,
        y + Math.cos(rotation) * size * 1.5,
    );

    this.lineTo(
        x - Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
        y + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
        x - Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
        y + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
        x - Math.sin(rotation) * size * 1.5,
        y + Math.cos(rotation) * size * 1.5,
    );

    this.fillStyle = 'rgb(255, 255, 255)';
    this.fill();
};

CanvasRenderingContext2D.prototype.drawCircle = function (x, y, radius) {
    this.beginPath();
    this.arc(x, y, radius, 0, 2.0 * Math.PI);
    this.fillStyle = 'rgb(255, 165, 0)';
    this.fill();
};

const simulation = new sim.Simulation();
const viewport = document.getElementById("viewport");
const ctxt = viewport.getContext("2d");
ctxt.fillStyle = "rgb(0, 0, 0";

function redraw() {
    let viewportWidth;
    let viewportHeight;
    if (window.innerWidth > window.innerHeight) {
        viewportHeight = window.innerHeight;
        viewportWidth = window.innerHeight;
    }
    else {
        viewportWidth = window.innerWidth;
        viewportHeight = window.innerWidth;
    }
    const viewportScale = window.devicePixelRatio || 2;
    viewport.width = viewportWidth * viewportScale;
    viewport.height = viewportHeight * viewportScale;
    ctxt.scale(viewportScale, viewportScale);
    ctxt.clearRect(0, 0, viewportWidth, viewportHeight);

    simulation.step();

    const world = simulation.world();

    for (const food of world.foods) {
        ctxt.drawCircle(
            food.x * viewportWidth,
            food.y * viewportHeight,
            (0.01 / 2.0) * viewportWidth,
        );
    }

    for (const animal of world.animals) {
        ctxt.drawTriangle(
            animal.x * viewportWidth,
            animal.y * viewportHeight,
            0.01 * viewportWidth,
            animal.rotation,
            );
    }

    requestAnimationFrame(redraw);
}
redraw();

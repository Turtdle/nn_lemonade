function Simulation() {
    this.images = null;
    this.context = null;
    this.time = 0;
    this.duration = 1000;
    this.frameRate = 4;

    this.onUpdate = null;
    this.onStep = null;
    this.onStop = null;

    this._requestId = -1;
    this._lastTime = -1;
    this._frameTime = 0;
    this.stopped = true;
    this.update = this.update.bind(this);

    this.customers = [];
    this.inPitcher = 0;
    this.soldOut = false;
    this.totalSold = 0;
    this.totalCustomers = 0;
}

Simulation.prototype.start = function() {
    this.stopped = false;
    this.time = 0;
    this._frameTime = 0;
    this.inPitcher = 0;
    this.soldOut = false;
    this.totalSold = 0;
    this.totalCustomers = 0;

    Customer.reset();

    this._requestId = requestAnimationFrame(this.update);
    this._lastTime = performance.now();

    this.refillPitcher();
    this.step();
};

Simulation.prototype.refillPitcher = function() {
    if (this.inPitcher === 0 &&
        state.data.lemons >= state.data.recipeLemons &&
        state.data.sugar >= state.data.recipeSugar) {
        this.inPitcher = 8 + state.data.recipeIce;
        state.data.lemons -= state.data.recipeLemons;
        state.data.sugar -= state.data.recipeSugar;
    }

    if (this.inPitcher === 0 || state.data.cups === 0 || state.data.ice < state.data.recipeIce)
        this.soldOut = true;
};

Simulation.prototype.buyOrPass = function() {
    var demand = ((state.data.temperatureFarenheit - 50) / 200 + (5 - this.weatherIndex) / 20) *
                 (((state.data.temperatureFarenheit / 4) - state.data.price) / (state.data.temperatureFarenheit / 4) + 1);
    if (state.data.repLevel < Math.random() * (state.data.repLevel - 500))
        demand = demand * state.data.reputation;
    demand *= (state.data.recipeLemons + 1) / 5;
    demand *= (state.data.recipeSugar + 4) / 8;

    for (var i = 0; i < this.customers.length; i++) {
        var customer = this.customers[i];
        if (customer.bubbleTime > 0)
            demand *= customer.bubble === 0 ? 1.3 : 0.5;
    }

    return (demand + random.uniform(-0.1, 0.1)) * 1.3;
};

Simulation.prototype.buyGlass = function(customer) {
    if (!this.soldOut && this.inPitcher > 0 &&
        state.data.cups > 0 && state.data.ice >= state.data.recipeIce) {
        this.inPitcher -= 1;
        state.data.ice -= state.data.recipeIce;
        state.data.cups -= 1;
        state.data.money += state.data.price;
        state.data.totalIncome += state.data.price;
        this.totalSold += 1;

        this.refillPitcher();

        if (this.giveRep() < 1) {
            var bubble = this.checkBubble();
            if (bubble > 0)
                customer.addBubble(bubble);
        } else if (Math.random() < 0.3)
            customer.addBubble(0);

        return true;
    } else {
        this.soldOut = true;
        return false;
    }
};

Simulation.prototype.giveRep = function() {
    var opinion = 0.8 + Math.random() * 0.4;
    opinion *= state.data.recipeLemons / 4;
    opinion *= state.data.recipeSugar / 4;
    opinion *= state.data.recipeIce / ((state.data.temperatureFarenheit - 50) / 5) + 1;
    opinion *= ((state.data.temperatureFarenheit - 50) / 5 + 1) / (state.data.recipeIce+4);
    opinion *= (state.data.temperatureFarenheit / 4 - state.data.price) / (state.data.temperatureFarenheit/4) + 1;
    opinion = Math.min(Math.max(opinion, 0), 2);
    state.data.reputation += opinion;
    state.data.repLevel++;
    return opinion;
};

Simulation.prototype.checkBubble = function() {
    // Yuck!
    if (state.data.recipeLemons < 4 || state.data.recipeSugar < 4)
        _reasons[2] = 1;
    else
        _reasons[2] = 0;

    // More ice!
    if (state.data.recipeIce < (state.data.temperatureFarenheit - 49) / 5)
        _reasons[1] = 1;
    else
        _reasons[1] = 0;

    // $$!
    if (state.data.price > state.data.temperatureFarenheit / 4)
        _reasons[0] = 1;
    else
        _reasons[0] = 0;

    var a = Math.floor(Math.random() * 3);
    return _reasons[a] === 1 ? a + 1 : 0;
};

Simulation.prototype.drawRain = function() {
    this.context.lineWidth = 1;
    this.context.strokeStyle = '#999';
    this.context.beginPath();

    var maxRain = random.integer(200, 400);
    for (var t = 0; t < maxRain; t++) {
        var x = Math.random() * 576;
        var y = Math.random() * 378;
        this.context.moveTo(x, y);
        this.context.lineTo(x + 2, y + 6);
    }

    this.context.stroke();
};

Simulation.prototype.update = function(now) {
    if (this.stopped)
        return;

    var elapsed = Math.min(0.1, (now - this._lastTime) / 1000);
    this._lastTime = now;
    this._frameTime += elapsed;

    if (this.onUpdate)
        this.onUpdate.call(null, this);

    if (this._frameTime >= 1/this.frameRate) {
        this.step();
        this._frameTime -= 1/this.frameRate;
    }

    if (this.time >= this.duration)
        this.stop();
    else
        this._requestId = requestAnimationFrame(this.update);
};

Simulation.prototype.step = function() {
    this.time += 1;

    if (Math.random() < 0.1)
        this.addCustomer();

    this.context.drawImage(this.images[WEATHER_TO_BACKGROUND[state.data.weather]], 0, 0);

    if (this.soldOut)
        this.context.drawImage(this.images['images/sold_out.png'], 220, 292);

    for (var i = 0; i < this.customers.length; i++)
        if (!this.customers[i].step()) {
            this.customers.splice(i, 1);
            i -= 1;
        }

    if (state.data.weather === 'rain')
        this.drawRain();

    if (this.onStep)
        this.onStep.call(null, this);
};

Simulation.prototype.addCustomer = function() {
    this.customers.push(new Customer(this));
    this.totalCustomers += 1;
};

Simulation.prototype.stop = function() {
    this.stopped = true;
    this.customers.length = 0;
    if (this._requestId !== -1) {
        cancelAnimationFrame(this._requestId);
        this._requestId = -1;
    }

    if (this.onStop)
        this.onStop.call(null, this);
};

Object.defineProperties(Simulation.prototype, {
    'weatherIndex': {
        'get': function() {
            return WEATHER_TO_NUMBER[state.data.weather];
        }
    }
});
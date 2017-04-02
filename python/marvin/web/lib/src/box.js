/*
* @Author: Brian Cherinka
* @Date:   2016-12-13 09:41:40
* @Last Modified by:   Brian Cherinka
* @Last Modified time: 2017-04-01 01:20:05
*/

// Using Mike Bostocks box.js code
// https://bl.ocks.org/mbostock/4061502

// This has been modified by me to accept data as a
// a list of objects in the format of
// data = [ {'value': number, 'title': string_name, 'sample': array of points}, ..]
// This allows to display box and whisker plots of an array of data
// and overplot a single value within this space

// Dec-13-2016 - converted to D3 v4

//jshint esversion: 6
'use strict';

function iqr(k) {
      return function(d, index) {
        let q1 = d.quartiles[0],
            q3 = d.quartiles[2],
            iqr = (q3 - q1) * k,
            i = -1,
            j = d.length;
        while (d[++i] < q1 - iqr);
        while (d[--j] > q3 + iqr);
        return [i, j];
      };
}

function boxWhiskers(d) {
  return [0, d.length - 1];
}

function boxQuartiles(d) {
  return [
    d3.quantile(d, .25),
    d3.quantile(d, .5),
    d3.quantile(d, .75)
  ];
}

function getTooltip() {
    let tooltip = d3.select('body').append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);
    return tooltip;
}

// Inspired by http://informationandvisualization.de/blog/box-plot
d3.box = function() {
  let width = 1,
      height = 1,
      duration = 0,
      domain = null,
      value = Number,
      whiskers = boxWhiskers,
      quartiles = boxQuartiles,
      showLabels = true,
      x1 = null,  // the x1 variable here represents the y-axis
      x0 = null, // the old y-axis
      tickFormat = null;

  let tooltip = getTooltip();

  // For each small multiple…
  function box(g) {
    g.each(function(d, i) {
      let origd = d;
      d = d.sample.map(value).sort(d3.ascending);
      let g = d3.select(this),
          n = d.length,
          min = d[0],
          max = d[n - 1];


      // Compute quartiles. Must return exactly 3 elements.
      let quartileData = d.quartiles = quartiles(d);
      let q10 = d3.quantile(d, .10);
      let q90 = d3.quantile(d, .90);
      let myiqr = quartileData[2]-quartileData[0];

      // compute the 2.5*iqr indices and values
      let iqr25inds = iqr(2.5).call(this, d, i);
      let iqr25data = iqr25inds.map(function(i) {return d[i];});

      // Compute whiskers. Must return exactly 2 elements, or null.
      let whiskerIndices = whiskers && whiskers.call(this, d, i),
          whiskerData = whiskerIndices && whiskerIndices.map(function(i) { return d[i]; });

      // Compute outliers. If no whiskers are specified, all data are "outliers".
      // We compute the outliers as indices, so that we can join across transitions!
      let outlierIndices = whiskerIndices
          ? d3.range(0, whiskerIndices[0]).concat(d3.range(whiskerIndices[1] + 1, n))
          : d3.range(n);

      // Compute the new x-scale.
      let q50 = quartileData[1];
      let zero = Math.max(iqr25data[1]-q50,q50-iqr25data[0]); //rescales the axis to center each plot on the median
      let diff = Math.min(max-whiskerData[1], min-whiskerData[0]);

      x1 = d3.scaleLinear()
          .domain([q50-zero, q50, q50+zero])
          .range([height, height/2, 0]);
          //.domain([min,max])
          //.range([height,0]);

      // Retrieve the old x-scale, if this is an update.
      x0 = this.__chart__ || d3.scaleLinear()
          .domain([0, Infinity])
          .range(x1.range());

      // Stash the new scale.
      this.__chart__ = x1;

      // Note: the box, median, and box tick elements are fixed in number,
      // so we only have to handle enter and update. In contrast, the outliers
      // and other elements are variable, so we need to exit them! Variable
      // elements also fade in and out.

      // Update center line: the vertical line spanning the whiskers.
      let center = g.selectAll("line.center")
          .data(whiskerData ? [whiskerData] : []);

      center.enter().insert("line", "rect")
          .attr("class", "center")
          .attr("x1", width / 2)
          .attr("y1", function(d) { return x0(d[0]); })
          .attr("x2", width / 2)
          .attr("y2", function(d) { return x0(d[1]); })
          .style("opacity", 1e-6)
        .transition()
          .duration(duration)
          .style("opacity", 1)
          .attr("y1", function(d) { return x1(d[0]); })
          .attr("y2", function(d) { return x1(d[1]); });

      center.transition()
          .duration(duration)
          .style("opacity", 1)
          .attr("y1", function(d) { return x1(d[0]); })
          .attr("y2", function(d) { return x1(d[1]); });

      center.exit().transition()
          .duration(duration)
          .style("opacity", 1e-6)
          .attr("y1", function(d) { return x1(d[0]); })
          .attr("y2", function(d) { return x1(d[1]); })
          .remove();

      // Update innerquartile box.
      let box = g.selectAll("rect.box")
          .data([quartileData]);

      box.enter().append("rect")
          .attr("class", "box")
          .attr("x", 0)
          .attr("y", function(d) { return x0(d[2]); })
          .attr("width", width)
          .attr("height", function(d) { return x0(d[0]) - x0(d[2]); })
        .transition()
          .duration(duration)
          .attr("y", function(d) { return x1(d[2]); })
          .attr("height", function(d) { return x1(d[0]) - x1(d[2]); });

      box.transition()
          .duration(duration)
          .attr("y", function(d) { return x1(d[2]); })
          .attr("height", function(d) { return x1(d[0]) - x1(d[2]); });

      // Update median line.
      let medianLine = g.selectAll("line.median")
          .data([quartileData[1]]);

      medianLine.enter().append("line")
          .attr("class", "median")
          .attr("x1", 0)
          .attr("y1", x0)
          .attr("x2", width)
          .attr("y2", x0)
        .transition()
          .duration(duration)
          .attr("y1", x1)
          .attr("y2", x1);

      medianLine.transition()
          .duration(duration)
          .attr("y1", x1)
          .attr("y2", x1);

      // Update whiskers.
      let whisker = g.selectAll("line.whisker")
          .data(whiskerData || []);

      whisker.enter().insert("line", "circle, text")
          .attr("class", "whisker")
          .attr("x1", 0)
          .attr("y1", x0)
          .attr("x2", width)
          .attr("y2", x0)
          .style("opacity", 1e-6)
        .transition()
          .duration(duration)
          .attr("y1", x1)
          .attr("y2", x1)
          .style("opacity", 1);

      whisker.transition()
          .duration(duration)
          .attr("y1", x1)
          .attr("y2", x1)
          .style("opacity", 1);

      whisker.exit().transition()
          .duration(duration)
          .attr("y1", x1)
          .attr("y2", x1)
          .style("opacity", 1e-6)
          .remove();

     // update datapoint circle
     if (origd.value) {

         let datapoint = g.selectAll('circle.datapoint')
            .data([origd.value]);

        datapoint.enter().append('circle')
            .attr('class', 'datapoint')
            .attr('cx', width/2)
            .attr('cy', (x1(origd.value)))
            .attr("r", 5)
            .style('fill', 'red')
            .on("mouseover", function(d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip .html('<b> Value: '+origd.value+'</b>')
                    .style("left", (d3.event.pageX) + "px")
                    .style("top", (d3.event.pageY - 28) + "px");
                })
            .on("mouseout", function(d) {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            });

    }

    // update title
     if (origd.title) {
        let title = g.selectAll('text.title')
            .data([origd.title]);
        title.enter().append('text')
        .text(origd.title)
        .attr('x', width/2)
        .attr('y', height)
        .attr('dy', 20)
        .attr('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .style('font-size', 15);
     }

      // Update outliers.
      let outlier = g.selectAll("circle.outlier")
          .data(outlierIndices, Number);

      outlier.enter().insert("circle", "text")
          .attr("class", "outlier")
          .attr("r", 5)
          .attr("cx", width / 2)
          .attr("cy", function(i) { return x0(d[i]); })
          .style("opacity", 1e-6)
        .on("mouseover", function(i) {
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html('<b> Value: '+d[i]+'</b>')
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY - 28) + "px");
            })
        .on("mouseout", function(d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        })
        .transition()
          .duration(duration)
          .attr("cy", function(i) { return x1(d[i]); })
          .style("opacity", 1);

      // outlier.transition()
      //     .duration(duration)
      //     .attr("cy", function(i) { return x1(d[i]); })
      //     .style("opacity", 1);

      // outlier.exit().transition()
      //     .duration(duration)
      //     .attr("cy", function(i) { return x1(d[i]); })
      //     .style("opacity", 1e-6)
      //     .remove();

      // Compute the tick format.
      let format = tickFormat || x1.tickFormat(8);

      // Update box ticks.
      let boxTick = g.selectAll("text.box")
          .data(quartileData);

      if (showLabels === true) {
          boxTick.enter().append("text")
              .attr("class", "box")
              .attr("dy", ".3em")
              .attr("dx", function(d, i) { return i & 1 ? 6 : -6; })
              .attr("x", function(d, i) { return i & 1 ? width : 0; })
              .attr("y", x0)
              .attr("text-anchor", function(d, i) { return i & 1 ? "start" : "end"; })
              .text(format)
            .transition()
              .duration(duration)
              .attr("y", x1);
      }

      boxTick.transition()
          .duration(duration)
          .text(format)
          .attr("y", x1);

      // Update whisker ticks. These are handled separately from the box
      // ticks because they may or may not exist, and we want don't want
      // to join box ticks pre-transition with whisker ticks post-.
      let whiskerTick = g.selectAll("text.whisker")
          .data(whiskerData || []);

      whiskerTick.enter().append("text")
          .attr("class", "whisker")
          .attr("dy", ".3em")
          .attr("dx", 6)
          .attr("x", width)
          .attr("y", x0)
          .text(format)
          .style("opacity", 1e-6)
        .transition()
          .duration(duration)
          .attr("y", x1)
          .style("opacity", 1);

      whiskerTick.transition()
          .duration(duration)
          .text(format)
          .attr("y", x1)
          .style("opacity", 1);

      whiskerTick.exit().transition()
          .duration(duration)
          .attr("y", x1)
          .style("opacity", 1e-6)
          .remove();
    });
    d3.timerFlush();
  }

box.overlay = function(x) {
    if (!arguments.length) return overlay;
    overlay = x;
    return overlay;
  };

box.x1 = function(x) {
    if (!arguments.length) return x1;
    return x1(x);
  };

box.x0 = function(x) {
    if (!arguments.length) return x0;
    return x0(x);
  };

  box.width = function(x) {
    if (!arguments.length) return width;
    width = x;
    return box;
  };

  box.height = function(x) {
    if (!arguments.length) return height;
    height = x;
    return box;
  };

  box.tickFormat = function(x) {
    if (!arguments.length) return tickFormat;
    tickFormat = x;
    return box;
  };

  box.duration = function(x) {
    if (!arguments.length) return duration;
    duration = x;
    return box;
  };

  box.domain = function(x) {
    if (!arguments.length) return domain;
    domain = x === null ? x : d3.functor(x);
    return box;
  };

  box.value = function(x) {
    if (!arguments.length) return value;
    value = x;
    return box;
  };

  box.tooltip = function(x) {
    if (!arguments.length) return tooltip;
    tooltip = x;
    return tooltip;
  };

  box.whiskers = function(x) {
    if (!arguments.length) return whiskers;
    whiskers = x;
    return box;
  };

  box.showLabels = function(x) {
    if (!arguments.length) return showLabels;
    showLabels = x;
    return box;
  };

  box.quartiles = function(x) {
    if (!arguments.length) return quartiles;
    quartiles = x;
    return box;
  };

  return box;
};

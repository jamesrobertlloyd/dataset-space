from pylab import * 

x = [900, 1200, 700, 1500, 850, 1410, 1000, 990,1150, 759, 815,800, 1400,1300, 1200, 1050, 1100]
y = [330, 575, 220, 800, 300, 745, 380, 360, 550, 260, 290, 289, 760,720, 600,450,500]

fit = polyfit(x,y,1)
fit_fn = poly1d(fit)
plot(x,y, 'ro', x, fit_fn(x), '-k')

xlabel('Size (area) of a London home in square feet')
ylabel('Cost of a London home in multiples of 1000 GBP')

xlim(680, 1550)
ylim(180, 830)

show()
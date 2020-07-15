import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

def plot_distribution(class_label):
  plt.figure(figsize=(10,5))
  plt.xlabel("confidence for CLASS-{}".format(class_label))
  plt.ylabel("# of examples")
  plt.xticks(np.arange(0, 1.0, 0.1))
  plt.grid()
  fig = plt.hist(conf[int(class_label)], np.arange(0, 1.1, 0.1))

def plot_figure(plot, x_label, x_values, y_label, y_values):
  plot.set_xlabel(x_label)
  plot.set_ylabel(y_label)
  plot.set_xlim(0, 1.0)
  plot.set_ylim(0, 1.0)
  plot.set_yticks(np.arange(0, 1.0, 0.05))
  plot.set_xticks(np.arange(0, 1.0, 0.05))
  plot.grid()
  plot.plot(x_values, y_values)

# printer = print_util.MessagePrinter()
# if CLASS_LABEL == '--select label--':
#   print(printer.GetPrintableErrorMessage("Please select a valid label from the dropdown list!"))
# else:
  
  
  plot_distribution(CLASS_LABEL)

  class_gt = (gt == int(CLASS_LABEL))
  precision, recall, thresholds = precision_recall_curve(class_gt, conf[int(CLASS_LABEL)])

  fig = plt.figure(figsize=(SCREEN_WIDTH, SCREEN_WIDTH / 3))
  plot1 = fig.add_subplot(1,3,1)
  plot_figure(plot1, "Recall", recall, "Precision", precision)

  plot2 = fig.add_subplot(1,3,2)
  # last element in precision is 1
  plot_figure(plot2, "Threshold", thresholds, "Precision", precision[:-1])

  plot3 = fig.add_subplot(1,3,3)
  # last element in recall is 0
  plot_figure(plot3, "Threshold", thresholds, "Recall", recall[:-1])

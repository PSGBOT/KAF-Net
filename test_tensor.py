from tensorboardX import SummaryWriter
writer = SummaryWriter("./logs/test")
writer.add_scalar("test", 1, 0)
writer.add_scalar("test", 2, 1)
writer.add_scalar("test", 3, 2)
writer.add_scalar("test", 4, 3)
writer.close()
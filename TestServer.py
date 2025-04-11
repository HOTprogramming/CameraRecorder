from networktables import NetworkTables

NetworkTables.initialize()
sd = NetworkTables.getTable("SmartDashboard")

while True:
    a = input("Enter Action: ")

    if a == "q":
        break
    elif a == "r":
        print("Recording")
        sd.putBoolean("Enabled", True)
    elif a == "s":
        print("Stopped Recording")
        sd.putBoolean("Enabled", False)
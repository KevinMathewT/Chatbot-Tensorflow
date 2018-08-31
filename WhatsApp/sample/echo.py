import time
from WhatsApp.webwhatsapi import WhatsAPIDriver
from WhatsApp.webwhatsapi.objects.message import Message

driver = WhatsAPIDriver(loadstyles=True, client='firefox')
print("Waiting for QR")
print("Bot started")

while True:
    time.sleep(3)
    print('Checking for more messages')
    for contact in driver.get_unread():
        for message in contact.messages:
            print(message.content)
            print(message.sender)

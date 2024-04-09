First generate output using the command generator for the robocup
(This worked for gpsr, I m not sure about egpsr)
Using regex I was able to remove every sentence that was unecessary. I used the following, replaced it with \n and removed the exceding \n characters

```re
([#]+(\s\w*\s\d+)?)|(.*\n(\t\s?[\w\s:'?,."\-!]*)+)
```

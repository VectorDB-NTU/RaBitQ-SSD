#ifndef UTILS_LOG_H_
#define UTILS_LOG_H_

#include <iostream>
#include <sstream>
#include <string>

// Minimal stream logger: LOG(level) << ...; prints "[file:line:LEVEL] ..."
// to stderr when the temporary is destroyed at the end of the statement.
//
// Each statement builds its message in a std::stringstream, which is too
// costly for high-frequency logging from many threads; this logger is meant
// for setup and error messages, not for hot paths.
class LogStream {
 public:
  LogStream(const char *file, int line, const char *level) {
    std::string filename = std::string(file);
    size_t pos = filename.find_last_of('/');
    if (pos != std::string::npos) {
      filename = filename.substr(pos + 1);
    }
    ss_ << "[" + filename + ":" + std::to_string(line) + ":" + level + "] ";
  }

  ~LogStream() {
    ss_ << std::endl;
    std::cerr << ss_.str();
  }

  template<class T>
  inline LogStream &operator<<(const T &t) {
    ss_ << t;
    return *this;
  }

 private:
  std::stringstream ss_;
};

#define LOG(level) \
  if (true)        \
  LogStream(__FILE__, __LINE__, #level)

#endif  // UTILS_LOG_H_
